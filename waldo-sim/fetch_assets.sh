#!/bin/sh
# Objects for waldo-sim: Google Scanned Objects in MuJoCo format (kevinzakka/mujoco_scanned_objects, a public sparse
# clone, ~20 MB) into assets/gso_repo. The robot model is in git (robot_model/g1_description); set FETCH_FULL_G1=1 to
# also rsync Unitree's whole g1_description from Waldo (needs access to Waldo).
set -e
cd "$(dirname "$0")"
mkdir -p assets
if [ -n "${FETCH_FULL_G1:-}" ]; then
  rsync -a ${WALDO_SSH:+-e "$WALDO_SSH"} waldo@100.120.182.66:soccerbot/assets/g1_description/ assets/g1_description/
fi
if [ ! -d assets/gso_repo ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/kevinzakka/mujoco_scanned_objects.git assets/gso_repo
fi
cd assets/gso_repo && git sparse-checkout set \
  models/Markings_Desk_Caddy models/Design_Ideas_Drawer_Store_Organizer models/Ecoforms_Cup_B4_SAN \
  models/3M_Vinyl_Tape_Green_1_x_36_yd models/Diamond_Visions_Scissors_Red models/Crayola_Crayons_24_count \
  models/Curver_Storage_Bin_Black_Small models/Big_Dot_Aqua_Pencil_Case \
  models/Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White models/Office_Depot_HP_950XL_Ink_Cartridge_Black_CN045AN
echo "assets ready"
