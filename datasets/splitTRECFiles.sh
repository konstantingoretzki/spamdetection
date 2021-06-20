cd trec07p
mkdir spam
mkdir ham
while read -r line
do
	label=$(echo $line | egrep -o "^[^ ]+")
	file=$(echo $line | egrep -o "[^ ]+$") 
	echo "Label: $label | file: $file | copyTo: ../$label/$(echo $file | egrep -o "[^/]+$")"
	mv $file ./$label/$(echo $file | egrep -o "[^/]+$")
	#break
done < ./full/index 

