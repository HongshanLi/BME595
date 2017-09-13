import logic_gates as lg

print("Testing AND with inputs (True, True), (True, False), (False, True), (False, False)")
and_ = lg.AND()
print(and_(True, True))
print(and_(True, False))
print(and_(False, True))
print(and_(False, False))

print("Testing NOT with inputs (True), (False)")
not_ = lg.NOT()
print(not_(True))
print(not_(False))

print("Testing OR with inputs (True, True), (True, False), (False, True), (False, False)")
or_ = lg.OR()
print(or_(True, True))
print(or_(True, False))
print(or_(False, True))
print(or_(False, False))


print("Testing XOR with inputs (True, True), (True, False), (False, True), (False, False)")
xor_ = lg.XOR()
print(xor_(True, True))
print(xor_(True, False))
print(xor_(False, True))
print(xor_(False, False))
