from explorer import explorer

for m in ['columns', 'office2', 'open_space_five', 'star_maze', 'topological_star_easier']:

    explorer('../../../scenarios/explorer.cfg', './sptm_maps/%s.wad'%(m), 0, './sptm_maps/%s.avi'%(m))
