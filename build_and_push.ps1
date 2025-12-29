param(
  [string]$Version = "v0.1.0"
)

docker build `
  -t "lukmaan434/factr-gpu:$Version" `
  -t "lukmaan434/factr-gpu:latest" `
  .

docker push "lukmaan434/factr-gpu:$Version"
docker push "lukmaan434/factr-gpu:latest"
