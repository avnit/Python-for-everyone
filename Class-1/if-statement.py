x = int(input("enter the marks obtained as an interger only ? "))
# Check if value greater than 90
if x > 90:
    print("Your Grade is A+")
# Check if value greater than 70
elif x > 70:
    print("Your Grade is A")
elif x > 65:
    print("Your Grade is B+")
elif x > 60:
    print("Your Grade is B")
elif x > 50:
    print("Your Grade is C")
elif x > 40:
    print("Your Grade is D")
else:
    print("Your Grade is F")
