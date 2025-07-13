import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class NEAXLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lateral_dim=None, lateral_strength=0.1):
        super().__init__()
        self.main = nn.Linear(in_dim, out_dim)
        self.lateral = nn.Linear(lateral_dim, out_dim) if lateral_dim else None
        self.lateral_strength = lateral_strength
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.main.weight)
        if self.lateral:
            nn.init.xavier_uniform_(self.lateral.weight)
            # Scale lateral weights to be smaller initially
            self.lateral.weight.data *= 0.1
    
    def forward(self, x, lateral_input=None):
        out = self.main(x)
        
        if self.lateral and lateral_input is not None:
            # Don't freeze lateral connections - let them learn!
            lateral_out = self.lateral(lateral_input)
            out = out + self.lateral_strength * lateral_out
        
        out = self.activation(out)
        return self.dropout(out)

class NEAXController:
    def __init__(self, min_epochs=10, performance_window=10, 
                 complexity_weight=0.1, entropy_threshold=0.3):
        self.min_epochs = min_epochs
        self.performance_window = performance_window
        self.complexity_weight = complexity_weight
        self.entropy_threshold = entropy_threshold
        
        # Track performance history
        self.loss_history = deque(maxlen=performance_window * 2)
        self.accuracy_history = deque(maxlen=performance_window * 2)
        self.last_expansion_epoch = 0
        self.expansion_cooldown = 15  # Minimum epochs between expansions
        
    def should_expand(self, loss, accuracy, entropy, current_params, epoch, total_epochs):
        if epoch < self.min_epochs:
            return False
            
        # Cooldown period after last expansion
        if epoch - self.last_expansion_epoch < self.expansion_cooldown:
            return False
            
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        if len(self.loss_history) < self.performance_window:
            return False
        
        # Check if there is a performance plateau
        recent_losses = list(self.loss_history)[-self.performance_window:]
        recent_accuracies = list(self.accuracy_history)[-self.performance_window:]
        
        # Calculate improvement trends
        loss_improvement = recent_losses[0] - recent_losses[-1]
        acc_improvement = recent_accuracies[-1] - recent_accuracies[0]
        
        # Check for plateau 
        loss_plateau = abs(loss_improvement) < 0.01
        acc_plateau = abs(acc_improvement) < 0.005
        
        # Calculate complexity penalty 
        max_reasonable_params = 200000  
        complexity_penalty = (current_params / max_reasonable_params) * self.complexity_weight
        
        # Expansion score
        entropy_score = max(0, entropy - self.entropy_threshold)
        plateau_score = 1.0 if (loss_plateau and acc_plateau) else 0.0
        progress_score = max(0, 1.0 - (epoch / total_epochs) * 0.5)  # Favor early expansion
        
        final_score = entropy_score + plateau_score + progress_score - complexity_penalty
        
        print(f"[Controller] Epoch {epoch}")
        print(f"  Entropy: {entropy:.3f} | Plateau: {plateau_score} | Progress: {progress_score:.3f}")
        print(f"  Complexity Penalty: {complexity_penalty:.3f} | Final Score: {final_score:.3f}")
        
        should_expand = final_score > 0.8
        
        if should_expand:
            self.last_expansion_epoch = epoch
            
        return should_expand

class NEAXNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, initial_hidden=64):
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.layers = nn.ModuleList()
        self.controller = NEAXController()
        
        # Start with a reasonable initial size
        self.layers.append(NEAXLayer(in_dim, initial_hidden, in_dim))
        self.output_layer = nn.Linear(initial_hidden, out_dim)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        original_input = x
        current = x
        
        # Forward through NEAX layers
        for layer in self.layers:
            current = layer(current, lateral_input=original_input)
        
        output = self.output_layer(current)
        return output
    
    def expand(self, entropy, base_neurons=64, max_neurons=128):
        current_complexity = len(self.layers)
        
        # Dynamic sizing based on entropy and position
        entropy_factor = min(2.0, max(0.5, entropy * 2))
        position_factor = 1.0 / (current_complexity + 1)  # Smaller layers as we go deeper
        
        neurons = int(base_neurons * entropy_factor * position_factor)
        neurons = min(max_neurons, max(16, neurons))  
        
        print(f"Adding layer with {neurons} neurons (entropy: {entropy:.3f})")
        
        # Determine input dimension for new layer
        if len(self.layers) == 0:
            in_dim = self.input_dim
        else:
            in_dim = self.layers[-1].main.out_features
        
        # Create new layer
        new_layer =NEAXLayer(
            in_dim=in_dim,
            out_dim=neurons,
            lateral_dim=self.input_dim,
            lateral_strength=0.1 / (len(self.layers) + 1)  # Weaker laterals as we go deeper
        )
        
        self.layers.append(new_layer.to(self.device))
        
        # Create new output layer
        old_output = self.output_layer
        new_output = nn.Linear(neurons, self.output_dim).to(self.device)
        
        nn.init.xavier_uniform_(new_output.weight)
        
        # Copy bias if possible
        if hasattr(old_output, 'bias') and old_output.bias is not None:
            new_output.bias.data = old_output.bias.data.clone()
        
        self.output_layer = new_output
        
        return neurons
    
    def progressive_freeze(self, freeze_ratio=0.5):
        num_layers = len(self.layers)
        if num_layers <= 1:
            return
        
        # Freeze only the oldest layers
        layers_to_freeze = int(num_layers * freeze_ratio)
        
        for i in range(layers_to_freeze):
            for param in self.layers[i].main.parameters():
                param.requires_grad = False
            # Keep lateral connections trainable
            if self.layers[i].lateral:
                for param in self.layers[i].lateral.parameters():
                    param.requires_grad = True
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

def train_improved_neax(model, train_loader, val_loader, device, max_epochs=50, lr=0.001):
    print("=== Training NEAX ===")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    results = {
        'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
        'params': [], 'entropy': []
    }
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (output.argmax(1) == target).sum().item()
            train_total += target.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                val_loss += loss.item()
                val_correct += (output.argmax(1) == target).sum().item()
                val_total += target.size(0)
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate entropy
        with torch.no_grad():
            sample_data = next(iter(val_loader))[0][:100].to(device)
            sample_data = sample_data.view(sample_data.size(0), -1)
            sample_output = model(sample_data)
            probs = F.softmax(sample_output, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        
        # Store results
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['train_loss'].append(avg_train_loss)
        results['val_loss'].append(avg_val_loss)
        results['params'].append(model.get_param_count())
        results['entropy'].append(entropy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Loss: {avg_val_loss:.4f} | Entropy: {entropy:.3f} | Params: {model.get_param_count():,}")
        
        # Check for expansion
        if model.controller.should_expand(avg_val_loss, val_acc, entropy, 
                                        model.get_param_count(), epoch, max_epochs):
            print("Expanding NEAX Network")
            
            # Progressive freezing
            model.progressive_freeze(freeze_ratio=0.3)
            
            # Expand the network
            new_neurons = model.expand(entropy)
            
            # Create new optimizer with lower LR for new parameters
            new_params = []
            old_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'layers.' + str(len(model.layers)-1) in name or 'output_layer' in name:
                        new_params.append(param)
                    else:
                        old_params.append(param)
            
            optimizer = torch.optim.Adam([
                {'params': old_params, 'lr': lr * 0.1},
                {'params': new_params, 'lr': lr}
            ], weight_decay=1e-5)
            
            # Fine-tune new layer for a few steps
            model.train()
            for ft_step in range(5):
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx >= 10:  
                        break
                        
                    data, target = data.to(device), target.to(device)
                    data = data.view(data.size(0), -1)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Unfreeze all for continued training
            model.unfreeze_all()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.5, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            print(f"Network expanded! New size: {model.get_param_count():,} parameters")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model, results

