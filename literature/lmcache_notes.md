# LMCache Notes

## Architecture
- KV cache chunk storage and retrieval for vLLM
- SemBlend wraps LMCacheConnectorV1 with semantic donor discovery
- Chunk-swap injection: contiguous prefix replacement (delta=0)
- CacheBlend mode: selective recomputation of mismatched tokens

## Key Integration Points
- `LMCacheConnectorV1`: vLLM's KV connector interface
- `SemBlendConnectorV1`: wraps LMCache, adds semantic matching
- Chunk storage: CPU DRAM (warm), can offload to disk (cold)

## TODO
- [ ] Read LMCache source to understand chunk format
- [ ] Understand CacheBlend selective recomputation API
- [ ] Map connector API for multi-model support
