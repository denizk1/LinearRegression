import matplotlib.pyplot as plt
from Data import Data
from LinearRegression import LinRes

def linRegPltShow(x,y,h,x_name,y_name,b,W,MSE,MAE):
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(x, y, 'go', label='Data')
    plt.plot(x, h, '--b', label='Liner Regression Function')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(x_name + " - " + y_name )
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(MSE, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(MAE, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(b, '-k')
    plt.xlabel('Epochs')
    plt.ylabel('b')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(W, '-k')
    plt.xlabel('Epochs')
    plt.ylabel('W')
    plt.grid()

    plt.show()

if __name__ == '__main__':

    ev2_data=Data("rvevfiyatlari.csv")
    x_train , y_train  = ev2_data.ReturnData()

    classifier=LinRes(x_train,y_train)
    h,b,W,MSE,MAE=classifier.train(verbose=True)

    linRegPltShow(x_train,y_train,h,"m2","fiyat",b,W,MSE,MAE)
