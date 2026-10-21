"""Fixed physical KV blocks, reference-counted forks and copy-on-write tails."""
import numpy as np


class BlockPool:
    def __init__(self, blocks, block_size, width):
        if min(blocks, block_size, width) < 1:
            raise ValueError("positive pool dimensions required")
        self.block_size, self.width = block_size, width
        self.k = np.zeros((blocks, block_size, width))
        self.v = np.zeros_like(self.k)
        self.refs = np.zeros(blocks, dtype=int)
        self.free = list(reversed(range(blocks)))
        self.copies = 0

    def allocate(self):
        if not self.free:
            raise MemoryError("KV block pool exhausted")
        block = self.free.pop()
        self.refs[block] = 1
        return block

    def release(self, block):
        if self.refs[block] <= 0:
            raise RuntimeError("double free")
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free.append(block)

    def snapshot(self):
        used = int(np.count_nonzero(self.refs))
        return dict(total_blocks=len(self.refs), used_blocks=used, free_blocks=len(self.free),
                    allocated_bytes=self.k.nbytes + self.v.nbytes,
                    live_block_bytes=used * self.block_size * self.width * 16,
                    copy_on_write_blocks=self.copies)


class PagedCache:
    def __init__(self, pool, offset=0):
        self.pool, self.offset = pool, offset
        self.width = pool.width
        self.blocks, self.length = [], 0

    def __len__(self):
        return self.length

    @property
    def end(self):
        return self.offset + self.length

    def _gather(self, array):
        if not self.blocks:
            return np.empty((0, self.width))
        return array[self.blocks].reshape(-1, self.width)[:self.length].copy()

    @property
    def k(self):
        return self._gather(self.pool.k)

    @property
    def v(self):
        return self._gather(self.pool.v)

    def append(self, k, v):
        k, v = np.asarray(k), np.asarray(v)
        if k.ndim != 2 or k.shape != v.shape or k.shape[1] != self.width:
            raise ValueError("matching [tokens, width] K/V required")
        if not len(k):
            return
        size = self.pool.block_size
        extra = (self.length + len(k) + size - 1) // size - len(self.blocks)
        cow = bool(self.length % size and self.pool.refs[self.blocks[-1]] > 1)
        # Check the entire append before changing any state.
        if extra + cow > len(self.pool.free):
            raise MemoryError("insufficient blocks for atomic append")
        if cow:
            old = self.blocks[-1]
            new = self.pool.allocate()
            self.pool.k[new] = self.pool.k[old]
            self.pool.v[new] = self.pool.v[old]
            self.pool.release(old)
            self.blocks[-1] = new
            self.pool.copies += 1
        self.blocks.extend(self.pool.allocate() for _ in range(extra))
        for i in range(len(k)):
            block, slot = divmod(self.length + i, size)
            self.pool.k[self.blocks[block], slot] = k[i]
            self.pool.v[self.blocks[block], slot] = v[i]
        self.length += len(k)

    def fork(self):
        child = PagedCache(self.pool, self.offset)
        child.blocks, child.length = self.blocks.copy(), self.length
        for block in child.blocks:
            self.pool.refs[block] += 1
        return child

    def truncate(self, length):
        if not 0 <= length <= self.length:
            raise ValueError("invalid rollback")
        keep = (length + self.pool.block_size - 1) // self.pool.block_size
        for block in self.blocks[keep:]:
            self.pool.release(block)
        self.blocks = self.blocks[:keep]
        self.length = length

    def release(self):
        self.truncate(0)


class PrefixStore:
    """Bounded LRU snapshots; a store belongs to one fixed model/position scheme.

    Lookup finds the longest exact token prefix. Snapshots and request branches
    share blocks, including a partial tail; appending triggers copy-on-write.
    """
    def __init__(self, pool, capacity=4):
        if capacity < 1:
            raise ValueError("positive prefix capacity required")
        from collections import OrderedDict
        self.pool, self.capacity = pool, capacity
        self.entries = OrderedDict()

    def put(self, tokens, cache):
        key = tuple(tokens)
        if not key or len(key) != len(cache) or cache.pool is not self.pool or cache.offset != 0:
            raise ValueError("prefix must match a position-zero cache from this pool")
        self.evict(key)
        while len(self.entries) >= self.capacity:
            self.evict(next(iter(self.entries)))
        self.entries[key] = cache.fork()

    def get(self, tokens):
        tokens = tuple(tokens)
        matches = [k for k in self.entries if tokens[:len(k)] == k]
        if not matches:
            return PagedCache(self.pool), 0
        key = max(matches, key=len)
        self.entries.move_to_end(key)
        return self.entries[key].fork(), len(key)

    def evict(self, key):
        cache = self.entries.pop(tuple(key), None)
        if cache is not None:
            cache.release()

    def clear(self):
        for key in list(self.entries):
            self.evict(key)
