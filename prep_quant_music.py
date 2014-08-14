from suffix_tree import SuffixTree
# http://www.daimi.au.dk/~mailund/suffix_tree.html

def main():
	file_path = '/media/internet/Shared/Uni/Bachelorarbeit/kdf_quantised/canon_quantized.txt'
	with open(file_path, "r") as myfile:
	    entire_string = myfile.read().translate(None, ' Keylength:')
 	longest_repeated_substring(entire_string, 20)

def create_file():
	with open('tempfile', "rw") as tmpf:
		tmpf.write("Uuuh test?")


def longest_repeated_substring(entire_string, freq_thresh=2):
	# https://en.wikipedia.org/wiki/Longest_repeated_substring_problem
	stree = SuffixTree(entire_string)
	count_descending_leaves(stree)
	depths = [(i.stringDepth, i.pathLabel) for i in stree.innerNodes if i.count >= freq_thresh]
	return max(depths)[1]
	#print longest.replace('[', '\n[')

def count_descending_leaves(stree):
	for n in stree.preOrderNodes:
		if not n.isLeaf:
			n.count = 0
	for l in stree.leaves:
		cur = l
		while cur.parent is not None:
			cur = cur.parent
			# print "Node {} increments {}.".format(l.pathLabel, cur.pathLabel)
			cur.count = cur.count + 1

if __name__ == '__main__':
	main()
	create_file()