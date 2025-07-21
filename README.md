1.install docker desktop

2.create your account

3.unable for wsl in it 

4.build docker image : docker build --platform linux/amd64 -t pdf-processor .

5.Run the Docker Container : 


docker run --rm \

  -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
  
  -v "$(pwd)/sample_dataset/outputs:/app/output" \
  
  -v "$(pwd)/sample_dataset/schema:/app/schema:ro" \
  
  --network none \
  
  pdf-processor
