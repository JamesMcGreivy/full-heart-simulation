input_mesh_dir="../data/instance_001"
input_mesh_header="instance_001"

output_mesh_dir="${input_mesh_dir}_lowres"
output_mesh_header="${input_mesh_header}_lowres"

export OMP_NUM_THREADS=80

mkdir -p ${output_mesh_dir}

input_mesh_path="${input_mesh_dir}/${input_mesh_header}"
output_mesh_path="${output_mesh_dir}/${output_mesh_header}"

# Check if output files exist and remove them
if ls ${output_mesh_path}* 1> /dev/null 2>&1; then
    rm ${output_mesh_path}*
fi

/work/submit/mcgreivy/openCARP/external/meshtool/meshtool resample mesh \
-msh=${input_mesh_path} \
-avrg=1000 \
-outmsh=${output_mesh_path} \
-ifmt=carp_txt \
-ofmt=carp_txt \
-tags=0 \

/work/submit/mcgreivy/openCARP/external/meshtool/meshtool extract surface \
-msh=${output_mesh_path} \
-surf=${output_mesh_path} \
-ifmt=carp_txt \
-ofmt=carp_txt

rm ${output_mesh_path}.neubc
rm ${output_mesh_path}.fcon
rm ${output_mesh_path}.surf.vtx
rm ${output_mesh_path}.surfmesh*

python cleanup_lowres.py --input_mesh_path ${input_mesh_path} --output_mesh_path ${output_mesh_path}