#!/bin/bash

round=4400
until [ $round -ge 60000 ] 
do
   python encode_Fmnist.py --line_number $round
   let round+=50
done