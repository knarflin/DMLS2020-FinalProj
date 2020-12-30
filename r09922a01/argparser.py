import sys

args = {'ep' : 100, 'size': 48, 'lr' : 0.01, 'bs' : 16, 'load' : None, 'save' : 'tmp.weight', 'show' : 1, 'site' : 0}
float_args = ['lr']
str_args = ['load', 'save']
args_parse = False # True if args have been parsed

def error_message(info) :
	print('Error :', info)
	quit()

def print_args() :
	if not args_parse :
		get_args()
	else :
		print('\nProgram :', sys.argv[0])
		print('\nArgument :')
		for arg in args :
			print('%-20s    '%arg, args[arg])
		print('')
	
def get_args() :
	global args_parse
	if args_parse :
		return args
	for index in range(1, len(sys.argv), 2) :
		arg = sys.argv[index][2:]
		if sys.argv[index][:2] == '--' and arg in args :
			if arg in float_args :
				args[arg] = float(sys.argv[index + 1])
			elif arg in str_args:
				args[arg] = sys.argv[index + 1]
			else :
				args[arg] = int(sys.argv[index + 1])
		else :
			error_message('Unrecognized argument : ' + sys.argv[index])
	args_parse = True
	if args['show']:
		print_args()
	return args

