LOCATIONS=("HYDRO" "SUCCULENT")
CAPTURE_DIR="/home/vuthanhdat/VisualLocalization/crocodl-benchmark/capture/"
CAPTURE_CROSS_VALID="/home/vuthanhdat/VisualLocalization/crocodl-benchmark/capture_cross_valid/"
DEVICES=("ios" "hl" "spot")
SESSIONS=("map" "query")



echo "You are running with parameters: "
echo "  Capture: ${CAPTURE_DIR}"
echo "  Capture_cross_valid: ${CAPTURE_CROSS_VALID}"
echo "  Locations: ${LOCATIONS[@]}"
echo "  Reference devices: ${DEVICES[@]}"

read -p "Do you want to continue? (y/n): " answer

if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Execution aborted."
    exit 1
fi


for LOCATION in "${LOCATIONS[@]}"; do
    for DEVICE in "${DEVICES[@]}"; do
        for SESSION in "${SESSIONS[@]}"; do
            # CAPTURE="${CAPTURE_DIR}/${LOCATION}/${SESSION}"
            # CROSS_VALID_CAPTURE="${CAPTURE_CROSS_VALID}/${LOCATION}/${SESSION}"
            # Do not remove or change this line if you intend to use automatic recall reading tool.
            echo "start create data cross valid for session: $SESSION scene: $LOCATION in capture cross valid: $CAPTURE_CROSS_VALID"
            DEST_DIR="${CAPTURE_CROSS_VALID}${LOCATION}/sessions/${DEVICE}_${SESSION}"
            if [ ! -d "$DEST_DIR" ]; then
                mkdir -p $DEST_DIR
                echo "$DEST_DIR has been created"
            else
                echo "$DEST_DIR already exist"
            
            fi

            python -m lamar.utils.get_data_cross_valid \
            --capture_dir "$CAPTURE_DIR"\
            --capture_cross_valid_dir "$CAPTURE_CROSS_VALID"\
            --location "$LOCATION"\
            --device "$DEVICE"\
            --session "$SESSION"\
            --images_txt "images.txt"\

            echo -e "create data cross valid completed for device: $DEVICE, session: $SESSION, scene: $LOCATION, in capture cross valid: $CAPTURE_CROSS_VALID"
        done
    done
done