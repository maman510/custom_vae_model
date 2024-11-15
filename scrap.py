     
    def fit(self, x, y, batch_size=1, epochs=1, verbose=1, learning_rate=0.001, validation_data=None, callbacks=None):
            # Custom fit method
            optimizer = self.optimizer(learning_rate=learning_rate)
            loss_fn = self.loss_fn
            
            # Number of steps per epoch
            num_batches = x.shape[0] // batch_size if batch_size else x.shape[0]
            
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                
                for step in range(num_batches):
                    start_idx = step * batch_size
                    end_idx = (step + 1) * batch_size
                    
                    # Get the current batch of data
                    x_batch = x[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]

                    print(tool_box.color_string('red', f"\n\n\toriginal x_batch shape: {x_batch.shape}\n"))
                   
                    x_batch = x_batch.astype('float32') / 255.0
                    x_batch = x_batch.reshape(-1, 28 * 28)
                    print(tool_box.color_string('red', f"\n\n\treshaped x_batch shape: {x_batch.shape}\n"))

                    with tf.GradientTape() as tape:
                        # Forward pass
                        logits = self(x_batch)
                        
                        # Compute loss
                        loss = loss_fn(y_batch, logits)
                    
                    # Compute gradients
                    grads = tape.gradient(loss, self.trainable_variables)
                    
                    # Apply gradients
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    
                    # Print the loss for each step (optional)
                    if verbose:
                        display = tool_box.color_string("yellow", f"Step {step+1}/{num_batches} - Loss: {loss.numpy():.4f}\n{step+1} image dimensions:\t{x_batch[0].shape}")
                        print(display)
                        time.sleep(2)
                
                # Validation logic (if provided)
                if validation_data:
                    x_val, y_val = validation_data
                    val_logits = self(x_val)
                    val_loss = loss_fn(y_val, val_logits)
                    print(f"Validation Loss: {val_loss.numpy():.4f}")
            