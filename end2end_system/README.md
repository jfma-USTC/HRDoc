# Python Environment
1. mmdet == 2.16.0
2. mmcv-full == 1.3.9
3. pdfplumber == 0.7.1
4. PyMuPDF == 1.19.6
4. torch == 1.6.0
5. torchvision == 0.7.0
6. transformers == 4.6.0

# Data Prepare
1. Place the ACL format pdfs into the path `pdf_parser/acl`
2. Execute stage1_data_prepare.py, you can see the raw image folder in `pdf_parser/acl` with the suffix `_vis`, and the raw annotation file `*.raw.json`, all information will be logged at `pdf_parser/acl_stage1.json`;
3. Execute stage2_data_prepare.py, you can see the merged annotation file `*.merged.json` in `pdf_parser/acl`, all information will be logged at `pdf_parser/acl_stage2.json`; 
4. Execute stage3_inference.py, the output predicted results are stored in path `strcut_recover/experiment/pred_result.json`
5. Execute stage4_data_postprocess.py, the output json files for each pdf will be stored in folder `generated_json`

# Files Structure
After all scripts are successfully executed, you will get the project file structure like follows:\
.\
│--- generated_json `(Stage 4)` \
│---│--- 2020.acl-main.2.json `(Stage 4)` \
│---│--- 2020.acl-main.3.json `(Stage 4)` \
│--- pdf_parser\
│---│--- acl\
│---│---│--- 2020.acl-main.2.merged.json `(Stage 2)` \
│---│---│--- 2020.acl-main.2.pdf `(Stage 0)` \
│---│---│--- 2020.acl-main.2.raw.json `(Stage 1)` \
│---│---│--- 2020.acl-main.2_vis `(Stage 1)` \
│---│---│---│--- XXX.png `(Stage 1)` \
│---│---│--- 2020.acl-main.3.merged.json `(Stage 2)` \
│---│---│--- 2020.acl-main.3.pdf `(Stage 0)` \
│---│---│--- 2020.acl-main.3.raw.json `(Stage 1)` \
│---│---│--- 2020.acl-main.3_vis `(Stage 1)` \
│---│---│---│--- XXX.png `(Stage 1)` \
│---│--- acl_stage1.json `(Stage 1)` \
│---│--- acl_stage2.json `(Stage 2)` \
│--- strcut_recover\
│---│--- experiment\
│---│---│--- pred_result.json `(Stage 3)`