"""Reconstruction of the model once the string is extracted"""
class Node:
   # position will give the position in options the next level was taken from
   def __init__(self, splitWidth, options, probabilities, position):
      # Left will the first one, right will be the last, as you would read it.
      self.linked = []
      if(position >= 0):
         # Set values for this node:
         # Left and right refer to the decision values, left being first, right second
         self.leftProb = float(probabilities[0][position])
         self.rightProb = float(probabilities[0][int(position + (len(probabilities[0]) / 2))])


         # Check to see if this is a leaf node:
         # Note, should not ever actually be < 1
         if len(options)  >= 2:
            # Set values for next nodes:
            self.nexts = options[1]
            # Holds the options for the next row of the tree
            # Create the next level of nodes
            for i in range(splitWidth[1]):
               self.linked.append(Node(splitWidth[1:], options[1:], probabilities[1:], i))
         else:
            self.nexts = []

      # Type == -1 means that it is the head node of the tree
      else:
         self.nexts = options[0]
         for i in range(splitWidth[0]):
            self.linked.append(Node(splitWidth, options, probabilities, i))
   


def test(head, data):
   # The two probabilities that will be compared in the end
   left = 1
   right = 1
   linked = len(head.nexts)
   while(len(linked) != 0):
      
      pass

   return left, right

def constructTree(model):
   # Seperate the model data into the rows it will be in by legal {} placements
   # model will be a string of the model data
   model = str(model)
   data = []
   splitWidth = []
   options = []
   probabilities = []
   for i in range(model.count('{')):
      start = model.find('{')
      end = model.find('}')
      data.append(model[start + 1:end + 1])
      model = model[end + 1:]

      # First string in parenthesis is the decision variable, ASSUMED BINARY
      splitWidth.append(data[i].count(data[i][1 : data[i].find("'", 2) + 1]))
      
      # Gather options for each level of the tree
      # trashData is used to avoid losing the last characters in the sliced string
      temp = []
      trashData = data[i][:]
      for j in range(splitWidth[i]):
         temp.append(trashData[trashData.find(',') + 2 : trashData.find(')')])
         temp[j] = temp[j].replace("'", "")
         trashData = trashData[trashData.find(',', trashData.find(':')) + 2 :]
      
      options.append(temp)
      
      # Pull out the probabilities for each value.
      temp = []
      trashData = data[i][:]
      for j in range((2 * splitWidth[i]) - 1):
         temp.append(float(trashData[trashData.find(':') + 1 : trashData.find(',', trashData.find(':'))]))
         trashData = trashData[trashData.find(',', trashData.find(':')) + 2 :]

      temp.append(float(data[i][data[i].rfind(':') + 1 : data[i].find('}')]))
      probabilities.append(temp)
   
   # Adding the decision values into the last spot in options
   temp = []
   temp.append(data[0][2 : data[0].find("'", 2)])
   temp.append(data[0][data[0].rfind("(") + 2 : data[0].rfind(',') - 1])
   options.append(temp)

   # At this point, the data is ready to be fed into the tree.
   # Need splitWidth, options, probabilities, position = -1
   head = Node(splitWidth, options[:-1], probabilities, -1)

   data = ["rainy", "hot", "high", "FALSE"]
   test(head, data)

