            dat = t[1:].reshape((28, 28))
            for k in range(28):
                for j in range(28): 
                    if dat[k][j]>0:
                        sys.stdout.write("#") 
                    else:
                        sys.stdout.write(".") 
                    sys.stdout.flush()
                sys.stdout.write("\n") 
            sleep(0.2)
            system('clear')
        
        for d in testing_data:
            prediction = p.predict(np.append([1],d[1:]))
            if prediction:
                print("YES!")
            else:
                print("NO!")