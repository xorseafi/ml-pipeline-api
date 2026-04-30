# ML Pipeline API (Commented Version)

## Steps to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
cd src
python train.py

3. Evaluate model:
python evaluate.py

4. Run API:
cd ../api
uvicorn main:app --reload

5. Open in browser:
http://127.0.0.1:8000/docs
