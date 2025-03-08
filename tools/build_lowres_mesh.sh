input_mesh_dir="../data/instance_001"
input_mesh_header="instance_001"

output_mesh_dir="${input_mesh_dir}_lowres"
output_mesh_header="${input_mesh_header}_lowres"

mkdir -p ${output_mesh_dir}

input_mesh_path="${input_mesh_dir}/${input_mesh_header}"
output_mesh_path="${output_mesh_dir}/${output_mesh_header}"

# Check if output files exist and remove them
if ls ${output_mesh_path}* 1> /dev/null 2>&1; then
    rm ${output_mesh_path}*
fi

meshtool resample mesh \
-msh=${input_mesh_path} \
-avrg=0.8 \
-outmsh=${output_mesh_path} \
-ifmt=carp_txt \
-ofmt=carp_txt

meshtool extract surface \
-msh=${output_mesh_path} \
-surf=${output_mesh_path} \
-ifmt=carp_txt \
-ofmt=carp_txt

rm ${output_mesh_path}.neubc
rm ${output_mesh_path}.fcon
rm ${output_mesh_path}.surf.vtx
rm ${output_mesh_path}.surfmesh*

python cleanup_lowres.py --input_mesh_path ${input_mesh_path} --output_mesh_path ${output_mesh_path}