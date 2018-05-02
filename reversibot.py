import json, sys, copy, time
startTime = time.time()
import numpy, datetime
import random, os, threading
import tensorflow as tf
import struct
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量
timeFormat = "%Y%m%d %H:%M:%S"
bestProb = 0.9
maxd = 2
FinalTurn = 53
maxThreads = 2
MaxTime = 45.6
LearningRate = 0.001
isReceiveData = True
isOnlineTrain = False
InitMaxValInt = 500
dataFile = "data/para2.data"
BuffSize = 12800
BatchSize = 128
Gamma = 0.98
FirstTrainNum = 0

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
                colors = ch2one_hot(color,len(games))
                #print(colors)
                evals = self.model.y.eval(session=self.sess,feed_dict={self.model.x:boards, self.model.keep_prob:1.0, self.model.colors_input:colors})               
                #print(evals)
                if self.color == -1:
                    evals = 2000 - evals
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
                if movlen == 0:
                    bmov,tmpval = self.alphaBeta(game,-color,-beta,-alpha,depth-1,isfinal,True)
                    return resmov,-tmpval
                
                boards = [g.board for g in gamelist]
                preboards = ch2FeedBoards(boards)
                colors = ch2one_hot(color,movlen)
                preevals = self.model.y.eval(session=self.sess,feed_dict={self.model.x:preboards, self.model.keep_prob:1.0, self.model.colors_input:colors})
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
        colors = ch2one_hot(self.color,moveNum)
        preevals = self.model.y.eval(session=self.sess,feed_dict={self.model.x:preboards, self.model.keep_prob:1.0, self.model.colors_input:colors})
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
            randf = random.random()
            #if self.isTrain:
                #print("rand:"+str(randint))
            if randf >  bestProb and self.color != lastWinner:
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
        self.all_units = [2 * 2 * self.con_units[-1],128,128,2]
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
            self.Q_colors = tf.matmul(hidden_dropout, self.Ws[self.all_layers-1]) + self.bs[self.all_layers-1]
            self.colors_input = tf.placeholder(tf.float32, [None,2])
            self.y = tf.reduce_sum(tf.multiply(self.Q_colors,self.colors_input), reduction_indices=1)
            self.y_ = tf.placeholder(tf.float32, [None])
            #计算均方差
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
def ch2one_hot(player_color,size):
    if player_color == -1:
        player_color = 0
    ans = numpy.zeros((size,2))
    for i in range(size):
        ans[i][player_color] = 1
    return ans

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

replay_buffer = deque()

def perceive(sess,model,game,color):
    global replay_buffer
    nowgame = copy.deepcopy(game)
    nextboards = []
    nextcolor = -color
    moves = nowgame.getValidMov(nextcolor)
    if len(moves) == 0:
        nextcolor = -nextcolor
        moves = nowgame.getValidMov(nextcolor)
    for mov in moves:
        nextgame = copy.deepcopy(nowgame)
        nextgame.place(mov[0],mov[1],nextcolor)
        nextboards.append(nextgame.board)
    
    replay_buffer.append((nowgame,color,nextcolor,nextboards))
    buffsize = len(replay_buffer)
    if buffsize > BuffSize:
        replay_buffer.popleft()
    if buffsize > 4*BatchSize:
        train_model(sess,model)
 
def train_model(sess,model):
    global replay_buffer
    buffsize = len(replay_buffer)
    train_time = time.time()
    databatch = random.sample(replay_buffer,BatchSize)
    boards = [d[0].board for d in databatch]
    colors = [d[1] for d in databatch]
    onehot_colors = numpy.zeros((BatchSize,2))
    for i in range(BatchSize):
        playcolor = 0 if colors[i] == -1 else 1
        onehot_colors[i][playcolor] = 1
    
    tvals = []
    for i in range(BatchSize):
        nowgame = databatch[i][0]
        nextcolor = databatch[i][2]
        nextboards = databatch[i][3]
        nextmovs = len(nextboards)
        if nextmovs == 0:
            tvals.append(1000 + 10 * (nowgame.blackPieceCnt - nowgame.whitePieceCnt))
        else:
            nextcolors = ch2one_hot(nextcolor,nextmovs)
            feedboards = ch2FeedBoards(nextboards)
            evals = model.y.eval(session=sess,feed_dict={model.x:feedboards, model.keep_prob:1.0, model.colors_input:nextcolors})
            evals = evals - 1000
            if nextcolor == 1:
                tvals.append(Gamma * numpy.max(evals) + 1000)
            else:
                tvals.append(Gamma * numpy.min(evals) + 1000)
    
    '''print(tvals)
    print(boards)
    print(onehot_colors)'''
    
    trainboards = ch2FeedBoards(boards)
    #Q_colors = model.Q_colors.eval(feed_dict={model.x:trainboards, model.keep_prob:1.0})
    #print(Q_colors)    
    sess.run(model.train_step,feed_dict={model.x:trainboards, model.colors_input:onehot_colors, model.y_:tvals, model.keep_prob:1.0})
    
    #Q_colors = model.Q_colors.eval(feed_dict={model.x:trainboards, model.keep_prob:1.0})
    #print(Q_colors)    
    #print(time.time()-train_time)
    
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
        '''if bestMov[0] >= 0:
            game.place(bestMov[0],bestMov[1],myColor)
            boards.append(copy.copy(game.board))
        offmoves = game.getValidMov(-myColor)
        if len(offmoves) == 1:
            game.place(offmoves[0][0],offmoves[0][1],-myColor)       
            boards.append(copy.copy(game.board))'''
        #print(len(boards))
        #print(boards)
        '''if game.isEnd() and game.getWinner() != myColor:
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
            isTrained = True'''
        res["debug"]["time"] = time.time() - startTime
        res["debug"]["isTrained"] = isTrained
        res["debug"]["isLoad"] = isLoad
        sys.stdout.write(json.dumps(res))
    else:
        sess = tf.InteractiveSession()
        model = createModel()
        #num_example = 10
        num = 1
        #his = []
        #his_y = []
        while True:
        #for n in range(num_example):
            print(num)
            roundTime = time.time()           
            game = Game()
            player1 = Player(1,game,model,sess,True)
            player2 = Player(-1,game,model,sess,True)
            turn = 1
            testboards = []
            testboards.append(copy.copy(game.board))
            #print(str(turn)+": ")
            #print(game.board)
            while not game.isEnd():
                bestMov,bestVal,_, = player1.searchPlace(turn)
                game.place(bestMov[0],bestMov[1],player1.color)
                perceive(sess,model,game,player1.color)
                testboards.append(copy.copy(game.board))
                #print(str(turn)+": ")
                #print(game.board)
                '''print(bestMov)
                print(bestVal)'''
                turn += 1
                if game.isEnd():
                    break
                bestMov,bestVal,_, = player2.searchPlace(turn)              
                game.place(bestMov[0],bestMov[1],player2.color)
                perceive(sess,model,game,player2.color)
                testboards.append(copy.copy(game.board))
                #print(str(turn)+": ")
                #print(game.board)
                '''print(bestMov)
                print(bestVal)'''
                turn += 1
            testboards = ch2FeedBoards(testboards)
            Q_colors = model.Q_colors.eval(feed_dict={model.x:testboards, model.keep_prob:1.0})
            print(Q_colors)
            #vals = model.y.eval(feed_dict={model.x:testboards, model.colors_input:testcolors, model.keep_prob:1.0})
            #print(vals)
            #exit(0)
            #print(boards)
            #boards.append(copy.copy(game.board))
            #print(vals)
            #print(boards)
            #print(vals)
            #print(len(boards))
            #his.append(blackboards)
            #his_y.append(10*(game.blackPieceCnt-game.whitePieceCnt))
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
            resfile = open("result.txt","a+")
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
        
        
