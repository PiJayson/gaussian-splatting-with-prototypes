#!/bin/bash

SERVER="gmum"
REMOTE_DIR="/home/z1180942/gaussian-splatting"
LOCAL_IMG_DIR="./img"

# Step 0: Clean img folder
echo "Cleaning img folder"
rm -rf "$LOCAL_IMG_DIR"/*
ssh "$SERVER" "cd $REMOTE_DIR && rm -rf img/"
ssh "$SERVER" "cd $REMOTE_DIR/output && rm -rf *"

# Step 1: Upload changed files
echo "Uploading changed files to server..."
rsync -avz --progress ./ "$SERVER":"$REMOTE_DIR"

# Step 2: Run sbatch learn.sh on the server
echo "Running sbatch learn.sh on server..."
ssh "$SERVER" "cd $REMOTE_DIR && sbatch learn.sh"

# Step 3: Clear error.log before listening
echo "Clearing error.log..."
ssh "$SERVER" "cd $REMOTE_DIR && > error.log"

# Step 4: Listen to error.log
echo "Listening to error.log (Press Ctrl+C to stop monitoring)..."
ssh "$SERVER" "cd $REMOTE_DIR && tail -f error.log" &

# Wait for user to stop monitoring with Ctrl+C
trap "echo 'Stopping log monitoring...'; kill %1" SIGINT
wait %1

# Step 5: Find the latest output folder and the largest iteration number
echo "Finding latest output folder..."
LATEST_FOLDER=$(ssh "$SERVER" "ls -td $REMOTE_DIR/output/*/ | head -n 1 | xargs -I {} basename {}")

echo "Latest folder found: $LATEST_FOLDER"

echo "Finding the latest iteration number..."
LATEST_ITERATION=$(ssh "$SERVER" "ls -d $REMOTE_DIR/output/$LATEST_FOLDER/point_cloud/iteration_*/ | awk -F'_iteration_' '{print \$2}' | sort -nr | head -n 1")

echo "Latest iteration found: $LATEST_ITERATION"

# Step 6: Download the latest point_cloud.ply
REMOTE_PLY_PATH="$REMOTE_DIR/output/$LATEST_FOLDER/point_cloud/iteration_$LATEST_ITERATION/point_cloud.ply"
LOCAL_PLY_PATH="point_cloud.ply"

echo "Downloading point_cloud.ply..."
rsync -avz --progress "$SERVER":"$REMOTE_PLY_PATH" "$LOCAL_PLY_PATH"

# Step 7: Download the img folder
echo "Downloading img folder..."
mkdir -p "$LOCAL_IMG_DIR"
rsync -avz --progress "$SERVER":"$REMOTE_DIR/img/" "$LOCAL_IMG_DIR"

echo "All tasks completed successfully!"
