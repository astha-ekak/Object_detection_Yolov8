from flask import Flask ,url_for ,request
from flask import render_template as rt
astha = Flask(__name__)      #instance 


@astha.route('/')  
def home():
    return rt('index.html')

@astha.route('/about')
def about():
    return rt('about.html') 


# @astha.route('/form',methods=['GET','POST'])
# def my_form():
#     if request.method=='GET':
#         return rt('form.html')
#     else:
#         return 'form Not submitted'
        
         

if __name__ == '__main__':
    astha.run(debug=True)