x=(int(input("please provide any input or 0 to terminate\n")))
while(x!=0):
    if((x%2==0)):
	    print(x,"is an even number")
    else:
	    print(x, " is an odd number")
    x = (int(input("please provide any input or 0 to terminate\n")))
    if(x==0):
	    print("You entered 0, program will exit now.")