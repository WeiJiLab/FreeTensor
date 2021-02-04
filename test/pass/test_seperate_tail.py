import ir

def test_basic():
	with ir.VarDef([
			("y1", (4,), "int32", "output", "cpu"),
			("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				y1[i] = 0
			with ir.Else():
				y1[i] = 1
			with ir.If(i < 2):
				y2[i] = 2
			with ir.Else():
				y2[i] = 3
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), "int32", "output", "cpu"),
			("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 2) as i:
			y1[i] = 0
			y2[i] = 2
		with ir.For("i", 2, 4) as i:
			y1[i] = 1
			y2[i] = 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_multiple_cond():
	with ir.VarDef([
			("y1", (5,), "int32", "output", "cpu"),
			("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 5) as i:
			with ir.If(i < 2):
				y1[i] = 0
			with ir.Else():
				y1[i] = 1
			with ir.If(i < 3):
				y2[i] = 2
			with ir.Else():
				y2[i] = 3
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (5,), "int32", "output", "cpu"),
			("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 2) as i:
			y1[i] = 0
			y2[i] = 2
		y1[2] = 1
		y2[2] = 2
		with ir.For("i", 3, 5) as i:
			y1[i] = 1
			y2[i] = 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_tiled():
	with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 3) as i:
			with ir.For("j", 0, 4) as j:
				with ir.If(4 * i + j < 10):
					y[4 * i + j] = 4 * i + j
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 2) as i:
			with ir.For("j", 0, 4) as j:
				y[4 * i + j] = 4 * i + j
		with ir.For("j", 0, 2) as j:
			y[8 + j] = 8 + j
	std = ir.pop_ast()

	assert std.match(ast)
