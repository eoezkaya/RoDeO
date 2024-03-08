from constraintFunction import ConstraintFunction

testConstraint = ConstraintFunction()
testConstraint.print()

testConstraint.expression = "x[0]**2 + sin(x[1]) - 2*x[3]"

testConstraint.generateCFunctionFromExpression()

print(testConstraint.CFunctionBody)

testConstraint.saveCFunctionToAFile()