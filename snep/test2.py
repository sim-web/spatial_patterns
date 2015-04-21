try:
	foo=0
	1/0
except:
	print 'failed'
try:
	print foo
except:
	pass