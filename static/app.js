class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            textField: document.querySelector('.chatbox__input'),
            chatMessageContainer: document.querySelector('.chatbox__messages') 
        }

        this.state = false;
        this.messages = [];
        // delay
        this.delay(1200).then(() => {
          this.messages.push({
              name: "Sam",
              message: "Welcome 🎉 to Andhra Teck League! 🔧🌐"
          });
          this.updateChatText(this.args.chatBox);
          return this.delay(2400); // Delay-2 seconds
      }).then(() => {
          this.messages.push({
              name: "Sam",
              message: "🎉 I am your virtual assistant🤖 here to help you. How can I assist you today?"
          });
          this.updateChatText(this.args.chatBox);
      });
  

          
    }
    

    display() {
        const {openButton, chatBox, sendButton} = this.args;
        // this.messages.push({
        //     name: "Sam",
        //     message:
        //       "🌐🔧 Welcome to Andhra Teck League! 🔧🌐\n\nGet ready to unlock a world of innovation and excitement with us! 🚀🎉",
        //   });
          this.updateChatText(chatBox);

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    async onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);
        this.updateChatText(chatbox);
        textField.value = '';

        await this.sendMessageWithDelay(chatbox, text1);
    } 

    async sendMessageWithDelay(chatbox, message) {
      let responses = await this.getBotResponses(message); // Wait for the responses
      for (const response of responses) {
          await this.delay(700); // Add a delay of 1000ms (1 second)

          let msg2 = { name: 'Sam', message: response };
          this.messages.push(msg2);
          this.updateChatText(chatbox);
      }
    }

  delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
  }
   
 
  async getBotResponses(message) {
    try {
        const response = await fetch("/predict", {
            method: 'POST',
            body: JSON.stringify({ message: message }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        });

        if (response.ok) {
            const responseData = await response.json();
            return [responseData.answer]; // Return the response from the server
        } else {
            console.error('Response not OK:', response.status);
            return ["Oops! Something went wrong on our end."];
        }
    } catch (error) {
        console.error('Error:', error);
        return ["Oops! Something went wrong on our end."];
    }
}

    updateChatText(chatbox) {
        var html = '';
      
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
    
}


const chatbox = new Chatbox();
chatbox.display();