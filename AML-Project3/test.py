from package import *

# x = np.loadtxt("train_x.csv", delimiter=",")
# y = np.loadtxt("train_y.csv", delimiter=",")
# # x = x.reshape(-1, 64, 64)
# # y = y.reshape(-1, 1)
# # # scipy.misc.imshow(x[0])
# #
# x_test = np.loadtxt("test_x.csv", delimiter=",")
# with open("pickle_dump_x.txt", "w")as fp:
#     pickle.dump(x, fp)
# with open("pickle_dump_y.txt", "w")as fp:
#     pickle.dump(y, fp)
# with open("pickle_test_x.txt", "w")as fp:
#     pickle.dump(x_test, fp)
#
# exit(0)

# with open("pickle_dump_x.txt", "r")as fp:
#     x = pickle.load(fp)
# with open("pickle_dump_y.txt", "r")as fp:
#     y = pickle.load(fp)
#
# plt.imshow(np.uint8(x[0]))
# plt.show()
a = np.array([0,0,0])
print np.shape(a)
# a = np.array([1.,0.,3.], dtype=int)
# print np.shape(a)
# n_values = np.max(a)+1
# print np.eye(n_values)[a]
# b = np.zeros((a.size, a.max()+1))
# b[np.arange(a.size), a] = 1
# print b
# i = 1
# print 'processing %d loop' %i


class_list =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
               21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]


a = np.empty(50000,)
a[2] = class_list.index(81)
print a[2]