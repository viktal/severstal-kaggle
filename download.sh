#!/bin/bash
kaggle competitions download -c severstal-steel-defect-detection
unzip -d data severstal-steel-defect-detection.zip
rm -rf severstal-steel-defect-detection.zip
