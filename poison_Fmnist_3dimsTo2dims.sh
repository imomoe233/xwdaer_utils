#!/bin/bash

round=0
step=10000
until [ $round -ge 10000 ] 
do
   python encode_Fmnist_3dimsTo2dims.py --line_number $round --step $step
   let round+=$step
done