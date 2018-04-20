import json, sys, copy, time
startTime = time.time()
import numpy, datetime
import random, os, threading
import tensorflow as tf
import struct

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量
timeFormat = "%Y%m%d %H:%M:%S"
bestProb = 1.0
maxd = 4
FinalTurn = 52
maxThreads = 2
MaxTime = 5.6
LearningRate = 0.0005
isReceiveData = True
isOnlineTrain = True
InitMaxValInt = 500
dataFile = "data/onlineData"

initData = {"conW":[[[[0]]]]}
#trainlag = 4
#traindis = [0.1,0.1,0.1,0.7]

class Game:
    
    def __init__(self):
        self.board = numpy.zeros((8,8), dtype = numpy.int)
        self.board[3][4] = self.board[4][3] = 1
        self.board[3][3] = self.board[4][4] = -1
        self.blackPieceCnt = 2
        self.whitePieceCnt = 2
    
    # 放置棋子，计算新局面
    def place(self, x, y, color, checkonly=False):
        if x < 0:
            return False
        board = self.board
        if board[x][y] != 0:
            return False
        valid = False
        for d in range(8):
            i = x + DIR[d][0]
            j = y + DIR[d][1]
            while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
                i += DIR[d][0]
                j += DIR[d][1]
            if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
                while True:
                    i -= DIR[d][0]
                    j -= DIR[d][1]
                    if i == x and j == y:
                        break
                    valid = True
                    if checkonly and valid:
                        return valid
                    board[i][j] = color
                    self.blackPieceCnt += color
                    self.whitePieceCnt -= color
        if valid:
            board[x][y] = color
            if color == 1:
                self.blackPieceCnt += 1
            else:
                self.whitePieceCnt += 1
        return valid
    
    def getValidMov(self, color):
        res = []
        for x in range(8):
            for y in range(8):
                if self.place(x,y,color,True):
                    res.append((x,y))
        return res
    
    def isEnd(self):
        if len(self.getValidMov(1)) == 0 and len(self.getValidMov(-1)) == 0:
            return True
        else:
            return False
    
    def getWinner(self):
        if self.blackPieceCnt > self.whitePieceCnt:
            return 1
        elif self.blackPieceCnt == self.whitePieceCnt:
            return 0
        else:
            return -1
    
    def __repr__(self):
        return "black:%d white:%d"%(self.blackPieceCnt,self.whitePieceCnt)    

lastWinner = -2        

class Player:
    
    def __init__(self,color,game=None,model=None,sess=None,isTrain=False):
        self.color = color
        self.isTrain = isTrain
        if game is None:
            self.game = Game()
        else:
            self.game = game
        self.model = model
        self.sess = sess
    
    class AlphaBetaThread(threading.Thread):
        
        def __init__(self,color,model,sess,args):
            threading.Thread.__init__(self)
            self.color = color
            self.model = model
            self.sess = sess
            self.args = args
            self.res = []
    
        def evalsGames(self,games,color,isfinal):
            if not isfinal:
                boards = [g.board for g in games]
                boards = ch2FeedBoards(boards)
                #print(boards)
                evals = self.sess.run(self.model.y, feed_dict={self.model.x:boards, self.model.y_:numpy.zeros((boards.shape[0],1)), self.model.keep_prob:1.0})               
                if self.color == -1:
                    evals = 2000 - evals
                #print(evals)
            else:
                #print(games)
                evals = numpy.zeros(len(games))
                for i in range(len(games)):
                    if self.color == 1:
                        evals[i] = 1000 + 10*(games[i].blackPieceCnt - games[i].whitePieceCnt)
                    else:
                        evals[i] = 1000 - 10*(games[i].blackPieceCnt - games[i].whitePieceCnt)
                #print(evals)
                #print(games)
                #exit(0)
            if self.color != color:
                evals = -evals
            evals = evals.flatten().tolist()
            #print(evals)
            maxidx = findMaxidx(evals)
            #print(evals[maxidx])
            #print(evals)
            return maxidx, evals[maxidx]
        
        def alphaBeta(self,game,color,alpha,beta,depth,isfinal,isNoMov=False):
            issame = 1
            if self.color != color:
                issame = -1
            if game.blackPieceCnt == 0:
                return (-1,-1),issame * (1000 - self.color*600)
            if game.whitePieceCnt == 0:
                return (-1,-1),issame * (1000 + self.color*600)
            moves = game.getValidMov(color)
            gamelist = []
            for mov in moves:
                newgame = copy.deepcopy(game)
                newgame.place(mov[0],mov[1],color)
                gamelist.append(newgame)
            
            #cnt = game.blackPieceCnt + game.whitePieceCnt
            movlen = len(moves)
            if depth == 0 or (isNoMov and movlen == 0) or (game.blackPieceCnt + game.whitePieceCnt == 63 and movlen != 0) or isTimeup():
                if movlen == 0:
                    gamelist.append(game)
                    moves.append((-1,-1))
                maxidx,maxeval = self.evalsGames(gamelist,color,isfinal)
                return moves[maxidx], maxeval
            else:
                resval = alpha
                resmov = [-1,-1]
                if len(moves) == 0:
                    bmov,tmpval = self.alphaBeta(game,-color,-beta,-alpha,depth-1,isfinal,True)
                    return resmov,-tmpval
                
                boards = [g.board for g in gamelist]
                preboards = ch2FeedBoards(boards)
                preevals = self.sess.run(self.model.y, feed_dict={self.model.x:preboards, self.model.y_:numpy.zeros((preboards.shape[0],1)), self.model.keep_prob:1.0})
                if self.color == -1:
                    preevals = 2000 - preevals
                preevals = issame * preevals
                preevals = preevals.flatten().tolist()
                evalmovlist = []
                for i in range(len(moves)):
                    evalmovlist.append((preevals[i],i))
                evalmovlist.sort(key=lambda x:x[0],reverse=True)

                isfound = False
                for i in range(len(evalmovlist)):
                    idx = evalmovlist[i][1]
                    if isfound:
                        bmov,tmpval = self.alphaBeta(gamelist[idx],-color,-resval-1,-resval,depth-1,isfinal)
                        tmpval = -tmpval
                        if tmpval > resval and tmpval < beta:
                            bmov,tmpval = self.alphaBeta(gamelist[idx],-color,-beta,-resval,depth-1,isfinal)
                            tmpval = -tmpval
                    else:
                        bmov,tmpval = self.alphaBeta(gamelist[idx],-color,-beta,-resval,depth-1,isfinal)
                        tmpval = -tmpval                   
                    if tmpval > resval:
                        resmov = moves[idx]
                        resval = tmpval
                        isfound = True
                    if resval >= beta:
                        return resmov, resval
                    #if resval > 1000 and i > 4:
                        #return resmov, resval
                return resmov, resval
        
        def run(self):
            self.res = self.alphaBeta(*self.args)
    
    def randSelSearch(self):
        moves = self.game.getValidMov(self.color)
        moveNum = len(moves)
        if moveNum == 0:
            return (-1, -1), -3000, 0
        gamelist = []
        for mov in moves:
            newgame = copy.deepcopy(self.game)
            newgame.place(mov[0],mov[1],self.color)
            gamelist.append(newgame)
        
        boards = [g.board for g in gamelist]
        preboards = ch2FeedBoards(boards)
        preevals = self.sess.run(self.model.y, feed_dict={self.model.x:preboards, self.model.y_:numpy.zeros((preboards.shape[0],1)), self.model.keep_prob:1.0})
        if self.color == -1:
            preevals = 2000 - preevals
        preevals = preevals.flatten().tolist()
        evalmovlist = []
        for i in range(moveNum):
            evalmovlist.append((preevals[i],i))
        evalmovlist.sort(key=lambda x:x[0],reverse=True)
        
        nowd = maxd - 1     
        bestVal = -3000
        while not isTimeup():
            resvals = []
            movi = 0
            tmpbestVal = -3000
            maxidx = 0
            while movi < moveNum and not isTimeup():
                movE = min(moveNum,movi+maxThreads)
                abthreads = []
                for i in range(movi,movE):
                    idx = evalmovlist[i][1]
                    abthread = self.AlphaBetaThread(self.color,self.model,self.sess,[gamelist[idx],-self.color,-3000,-tmpbestVal,nowd,False])
                    abthreads.append((abthread,idx))
                for t in abthreads:
                    t[0].start()
                for t in abthreads:
                    t[0].join()
                    resMov,val = t[0].res #self.alphaBeta(newgame,-self.color,-2000,2000,maxd-1,False)
                    val = -val
                    if val > tmpbestVal:
                        tmpbestVal = val
                        maxidx = t[1]
                    resvals.append(val)
                movi += maxThreads
            if isTimeup() and nowd != maxd - 1:
                break
            random.seed(time.time())
            randint = random.randint(1,100)
            #if self.isTrain:
                #print("rand:"+str(randint))
            if randint > int(100 * bestProb) and self.color != lastWinner:
                #print(self.color)
                maxidx = random.randint(0,len(moves)-1)
            bestMov = moves[maxidx]
            bestVal = tmpbestVal
            nowd += 1
            '''print(evalmovlist)
            print(resvals)
            print(moves)
            print(maxidx)
            print(nowd)
            print(bestVal)'''
            if self.isTrain:
                #print(resvals)
                #print(moves)
                break           
        return bestMov, bestVal, nowd
    
    def searchPlace(self,turn,isrand=False):
        global startTime
        if isrand:
            moves = self.game.getValidMov(self.color)
            if len(moves) == 0:
                return (-1, -1), -3000, 0
            return random.choice(moves),-3000,0
        else:
            if self.isTrain:
                startTime = time.time()
            if turn < FinalTurn:
                return self.randSelSearch()                   
            else:
                '''if turn < FinalTurn:
                    abthread = self.AlphaBetaThread(self.color,self.model,self.sess,[self.game,self.color,-2000,2000,maxd,False])
                    abthread.start()
                    abthread.join()
                    bestMov,bestVal = abthread.res#self.alphaBeta(self.game,self.color,-2000,2000,maxd,False)
                else:'''
                nowd = 64-FinalTurn
                abthread = self.AlphaBetaThread(self.color,self.model,self.sess,[self.game,self.color,-3000,3000,nowd,True])
                abthread.start()
                abthread.join()
                bestMov,bestVal = abthread.res 
                #bestMov,bestVal = self.alphaBeta(self.game,self.color,-2000,2000,62-FinalTurn,True)
            return bestMov, bestVal, nowd
    
    def playTurn(self,turn,isrand=False):
        bestMov,bestVal,_, = self.searchPlace(turn,isrand)
        print("Val:"+str(bestVal))
        print(bestMov)
        game.place(bestMov[0],bestMov[1],self.color)
        if bestMov[0] != -1:
            return bestMov, bestVal
        else:
            return bestMov, bestVal
 
class EvalModel:
    
    def __init__(self):
        self.learning_rate = LearningRate
        self.conWidth = 3
        self.con_units = [1,32,64]
        self.conW = []
        self.conb = []
        #self.all_units = [64,64,32,16,1]
        self.all_units = [2 * 2 * self.con_units[-1],128,128,1]
        #print(self.all_units[0])
        self.all_layers = len(self.all_units) - 1
        self.Ws = []
        self.bs = []

        with tf.name_scope("EvalModel") as name_scope:
        #with tf.device("/cpu:0"):
            self.x = tf.placeholder(tf.float32, [None, 8, 8, 1])
            self.keep_prob = tf.placeholder(tf.float32)
            pool = self.x #tf.reshape(self.x,(-1,8,8,1))
            for i in range(1,len(self.con_units)):
                if "updateTime" in initData.keys(): 
                    self.conW.append(tf.Variable(initData["conW"][i-1]))
                    self.conb.append(tf.Variable(initData["conb"][i-1]))
                else:
                    self.conW.append(tf.Variable(tf.truncated_normal([self.conWidth, self.conWidth, self.con_units[i-1], self.con_units[i]], stddev=0.1),name="conv_W"+str(i)))
                    self.conb.append(tf.Variable(tf.zeros([self.con_units[i]]),name="conv_b"+str(i)))
                conv = tf.nn.conv2d(pool, self.conW[i-1], strides=[1, 1, 1, 1], padding='SAME')
                relu = tf.nn.relu(conv + self.conb[i-1])
                #print(relu.shape)
                pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #print(pool.shape)
            
            hidden_dropout = tf.reshape(pool,(-1, 2*2*self.con_units[-1]))
            #hidden_dropout = self.x
            for i in range(self.all_layers):
                if "updateTime" in initData.keys(): 
                    self.Ws.append(tf.Variable(initData["fcW"][i]))
                    self.bs.append(tf.Variable(initData["fcb"][i]))
                else:
                    self.Ws.append(tf.Variable(tf.truncated_normal([self.all_units[i], self.all_units[i+1]],mean=0.1, stddev=0.1),name="fc_W"+str(i)))
                    self.bs.append(tf.Variable(tf.zeros([self.all_units[i+1]]),name="fc_b"+str(i)))                
            
            for i in range(self.all_layers - 1):
                hidden = tf.nn.relu(tf.matmul(hidden_dropout,self.Ws[i])+self.bs[i])
                hidden_dropout = tf.nn.dropout(hidden, self.keep_prob)
            self.y = tf.matmul(hidden_dropout, self.Ws[self.all_layers-1]) + self.bs[self.all_layers-1]
            self.y_ = tf.placeholder(tf.float32, [None, self.all_units[self.all_layers]])
            #计算均方差
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

def findMaxidx(vals):
    maxidx = -1
    max = -10000
    for i in range(len(vals)):
        if vals[i] > max:
            max = vals[i]
            maxidx = i
    return maxidx
        
def isTimeup():
    return time.time() - startTime >= MaxTime
        
def createModel():
    # Init model
    model = EvalModel()
    tf.global_variables_initializer().run()
    return model
   
# 处理输入，还原棋盘
def initGame(game,fullInput):
    boards = []
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    myColor = 1
    nowturn = 0
    boards.append(copy.copy(game.board))
    if requests[0]["x"] >= 0:
        myColor = -1
        game.place(requests[0]["x"], requests[0]["y"], -myColor)
        boards.append(copy.copy(game.board))
    else:
        nowturn -= 1
    turn = len(responses)
    nowturn += 2*(turn+1)
    for i in range(turn):
        if responses[i]["x"] >= 0: 
            game.place(responses[i]["x"], responses[i]["y"], myColor)        
            boards.append(copy.copy(game.board))  
        if requests[i+1]["x"] >= 0:
            game.place(requests[i + 1]["x"], requests[i + 1]["y"], -myColor)             
            boards.append(copy.copy(game.board))
    return myColor,game,nowturn,boards

def ch2FeedBoards(boards):   
    return numpy.array(boards).reshape(-1,8,8,1) + 2
    
def train_model(sess,model,boards,vals):
    max_epoch = 5
    ys = numpy.array(vals).reshape((-1,1))
    #print(ys)
    evals = sess.run(model.y, feed_dict={model.x:boards, model.y_:numpy.zeros(ys.shape), model.keep_prob:1.0})
    #print(evals)
    if (evals[0][0] > 1000 + InitMaxValInt and ys[-1] > 1000) or (evals[0][0] < 1000 - InitMaxValInt and ys[-1] < 1000):
        return
    #print(evals-ys)
    #print(" ")
    sumdis = numpy.sum((evals-ys)*(evals-ys))
    if sumdis > 1000000:
        max_epoch = 15
    elif sumdis > 250000:
        max_epoch = 10
    #print(" ")
    trainboards = boards[:]
    trainys = ys[:]
    for epoch in range(max_epoch):
        #tmpys = [ys[-1]]
        #for i in range(trainboards.shape[0]-1,-1,-1):
        sess.run(model.train_step,feed_dict={model.x:trainboards, model.y_:trainys, model.keep_prob:1.0})
        #tmpys = sess.run(model.y,feed_dict={model.x:[trainboards[i]], model.y_:[[0]], model.keep_prob:1.0})
        #trainys = numpy.vstack([tmpys[1:],ys[-1]])
        #for i in range(8):
            #trainys[-i] = ys[-i]
        #print(trainys)
            #tmpy = sess.run(model.y,feed_dict={model.x:[trainboards[i]], model.y_:[[0]], model.keep_prob:1.0})
            #print(tmpy)
    #testtime1 = time.time()        
    Aevals = sess.run(model.y, feed_dict={model.x:boards, model.y_:numpy.zeros(ys.shape), model.keep_prob:1.0})
    #print(Aevals)
    #print(time.time()-testtime1)
    #print(Aevals-evals)
    #print(sumdis)
    #print(max_epoch)
    #print(numpy.sum((Aevals-ys)*(Aevals-ys)))
    #print(Aevals[0][0])
    return Aevals
    
if __name__ == '__main__': 
    game = Game()
    if isReceiveData:
        try:
            if not isOnlineTrain:
                datafile = open(dataFile,"rb")
                tmpdata = datafile.read().decode()               
                datafile.close()
            else:
                tmpdata = ""
                for i in range(4):
                    datafile = open(dataFile+str(i),"rb")
                    tmpdata += datafile.read().decode()
                    datafile.close()                    
            initData = json.loads(tmpdata)
        except Exception as err:
            print(err)
            pass
    #print(initData["conW"][0][0][0][0])
    if len(sys.argv) < 2:
        fullInput = json.loads(input())
        isLoad = initData["conW"][0][0][0][0]
        #print(isLoad)
        sess = tf.InteractiveSession()
        model = createModel()
        myColor,game,nowturn,boards = initGame(game,fullInput)
        #print(boards)
        #print(game.board)
        myPlayer = Player(myColor,game,model,sess)
        bestMov,bestVal,nowd = myPlayer.searchPlace(nowturn)
        res = {"response": {"x": bestMov[0], "y": bestMov[1]}}
        #res["data"] = json.dumps(initData)
        res["debug"] = {}
        
        res["debug"]["val"] = bestVal
        res["debug"]["nowd"] = nowd+1
        
        isTrained = False
        if bestMov[0] >= 0:
            game.place(bestMov[0],bestMov[1],myColor)
            boards.append(copy.copy(game.board))
        offmoves = game.getValidMov(-myColor)
        if len(offmoves) == 1:
            game.place(offmoves[0][0],offmoves[0][1],-myColor)       
            boards.append(copy.copy(game.board))
        #print(len(boards))
        #print(boards)
        if game.isEnd() and game.getWinner() != myColor:
            trainboards = ch2FeedBoards(boards)
            #winner = game.getWinner()
            evals = sess.run(model.y, feed_dict={model.x:trainboards, model.y_:numpy.zeros((trainboards.shape[0],1)), model.keep_prob:1.0})
            evals = evals.flatten().tolist()
            #evals = numpy.hstack([evals[4:], evals[-4:]])
            cntScore = 1000 + 10 * (game.blackPieceCnt - game.whitePieceCnt)
            nVal = len(evals)
            for i in range(nVal):
                if nVal - i <= 11:
                    evals[i] = cntScore
                else:
                    evals[i] = evals[i+3]
            evals[0] = 1000
            evals = train_model(sess,model,trainboards, evals)
            data = {}
            ret = sess.run(model.conW)
            data["conW"] = [w.tolist() for w in ret]
            ret = sess.run(model.conb)
            data["conb"] = [b.tolist() for b in ret]
            ret = sess.run(model.Ws)
            data["fcW"] = [w.tolist() for w in ret]
            ret = sess.run(model.bs)
            data["fcb"] = [b.tolist() for b in ret]
            data["updateTime"] = datetime.datetime.now().strftime(timeFormat)
            res["debug"]["evals"] = evals.flatten().tolist()
            output = json.dumps(data)
            splitlen = int(len(output)/4) + 1
            output = [output[i:i+splitlen] for i in range(0,len(output),splitlen)]
            for i in range(4):
                wbfile = open(dataFile+str(i),"wb")          
                wbfile.write(output[i].encode(encoding='utf-8'))
                wbfile.close()
            isTrained = True
        res["debug"]["time"] = time.time() - startTime
        res["debug"]["isTrained"] = isTrained
        res["debug"]["isLoad"] = isLoad
        sys.stdout.write(json.dumps(res))
    else:
        sess = tf.InteractiveSession()
        model = createModel()
        #num_example = 200
        num = 1
        #his = []
        #his_y = []
        while True:
        #for n in range(num_example):
            print(num)
            roundTime = time.time()
            resfile = open("result.txt","a+")
            game = Game()
            boards = []
            vals = []
            boards.append(copy.copy(game.board))
            vals.append(1000.0)
            player1 = Player(1,game,model,sess,True)
            player2 = Player(-1,game,model,sess,True)
            turn = 1
            #print(str(turn)+": ")
            #print(game.board)
            isrand = True
            while not game.isEnd():
                bestMov,bestVal,_, = player1.searchPlace(turn)
                if bestMov[0] >= 0:
                    game.place(bestMov[0],bestMov[1],player1.color)
                    boards.append(copy.copy(game.board))
                    vals.append(bestVal)
                #print(str(turn)+": ")
                #print(game.board)
                '''print(bestMov)
                print(bestVal)'''
                turn += 1    
                bestMov,bestVal,_, = player2.searchPlace(turn)
                if bestMov[0] >= 0:                   
                    game.place(bestMov[0],bestMov[1],player2.color)
                    boards.append(copy.copy(game.board))
                    vals.append(2000-bestVal)
                #print(str(turn)+": ")
                #print(game.board)
                '''print(bestMov)
                print(bestVal)'''
                turn += 1
            #print(boards)
            #boards.append(copy.copy(game.board))
            blackboards = ch2FeedBoards(boards)
            '''cntScore = 1000 + 10 * (game.blackPieceCnt-game.whitePieceCnt)
            vals = sess.run(model.y, feed_dict={model.x:blackboards, model.y_:numpy.zeros((blackboards.shape[0],1)), model.keep_prob:1.0})
            vals = vals.flatten().tolist()
            for i in range(1,8):
                vals[-i] = cntScore
            tmpVals = []
            for i in range(trainlag):
                tmpVals.append(numpy.hstack([vals[i+1:],vals[-i-1:]]))
                #print(tmpVals[i])
            vals = numpy.zeros(len(vals))
            for i in range(trainlag):
                vals += traindis[i] * tmpVals[i]'''
            #print(vals)
            #print(boards)
            #print(vals)
            #print(len(boards))
            #his.append(blackboards)
            #his_y.append(10*(game.blackPieceCnt-game.whitePieceCnt))
            train_model(sess,model,blackboards,vals)
            if num % 10 == 0:
                wbfile = open(dataFile,"wb")
                data = {}
                res = sess.run(model.conW)
                data["conW"] = [w.tolist() for w in res]
                res = sess.run(model.conb)
                data["conb"] = [b.tolist() for b in res]
                res = sess.run(model.Ws)
                data["fcW"] = [w.tolist() for w in res]
                res = sess.run(model.bs)
                data["fcb"] = [b.tolist() for b in res]
                data["updateTime"] = datetime.datetime.now().strftime(timeFormat)
                wbfile.write(json.dumps(data).encode(encoding='utf-8'))            
                #initData = json.loads(json.dumps(data))
                wbfile.close()
            #print(boards.shape)
            
            winner = game.getWinner()
            lastWinner = winner
            print(winner)
            print(game.blackPieceCnt)
            print(game.whitePieceCnt)
            resfile.write(str(num)+" winner: "+str(winner)+" black:"+str(game.blackPieceCnt)+" white:"+str(game.whitePieceCnt)+"\n")
            resfile.close()
            #print(boards)
            print(time.time()-roundTime)
            num += 1
            #if n >= 400:
                #time.sleep(2)
        
        '''winS = []
        loseS = []
        for i in range(len(his)):
            sum = 0
            for board in his[i]:
                evals = sess.run(model.y, feed_dict={model.x:[board], model.y_:[[0]], model.keep_prob:1.0})
                print(evals)
                sum += evals[0][0]
            print(his_y[i])
            print(sum)
            if(his_y[i] > 0):
                winS.append(sum)
            elif his_y[i] < 0:
                loseS.append(sum)
            #time.sleep(2)
        print(winS)
        print(loseS)
        print("winS mean: "+str(numpy.mean(winS)))
        #print(loseS)
        print("loseS mean: "+str(numpy.mean(loseS)))'''
        
        