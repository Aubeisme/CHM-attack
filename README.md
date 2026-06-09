# CHM
PyTorch implementation of "Context-aware Hide-and-Mask Attack on Active Speaker Detection"

# Setup
1. ```
   conda env create -f AttackASD/ASDs/LoCoNet_ASD/environment.yml
   conda activate loconet
   ```

2. Set dataset path via environment variable:
   ```
   export DATA_PATH_AVA=/path/to/AVADataPath      # for AVA dataset
   export DATA_PATH_UNITALK=/path/to/UniTalk       # for UniTalk dataset
   ```

   Your folder structure should look like this:
   ```
   {DATA_PATH_AVA}/
   └── clips_audios/val/{video_id}/{entity_id}.wav
   └── clips_videos/val/{video_id}/{entity_id}/{timestamp}.jpg
   └── csv/
       ├── val_loader.csv
       └── val_orig.csv
   ```

3. Run:
   ```
   # Single-speaker models (LRASD, LightASD, TalkNet)
   python CHM.py --modelName LRASD --DEVICE cuda:0 --evaNum 1000 --datatype AVA

   # Multi-speaker model (LoCoNet_ASD), run from parent directory
   python main.py --attack CHM_FINAL --modelName LoCoNet_ASD --DEVICE cuda:0 --evaNum 1000 --datatype AVA
   ```
