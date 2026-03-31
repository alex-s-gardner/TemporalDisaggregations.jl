using ReTestItems
nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "2"))
runtests(dirname(@__DIR__); nworkers, nworker_threads=1, testitem_timeout=300)
