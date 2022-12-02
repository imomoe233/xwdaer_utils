#!/bin/bash

round=50
until [ $round -ge 10000 ] 
do
   python encode_Fmnist.py --line_number $round
   let round+=50
done