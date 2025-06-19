## Capsule spec (v0)

| field       | dtype            | bytes | note                              |
|-------------|------------------|-------|-----------------------------------|
| id          | uint64           | 8     | global unique                     |
| vec         | int8[dim]        | d     | bipolar or ternary hypervector    |
| last_used   | uint32           | 4     | unix-epoch seconds                |
| edge_ids    | uint64[k]        | 8k    | neighbour capsule ids (sparse)    |
| edge_wts    | int16[k]         | 2k    | signed 16-bit weights             |

Default `dim = 2048`, `k \u2264 5` hot edges.
