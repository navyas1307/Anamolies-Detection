class TransactionMonitor {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000';
        this.streamInterval = null;
        this.isStreaming = false;
        this.transactions = [];
        this.maxRows = 100;
        this.isPaused = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.startStreaming();
        this.updateSummaryCards();
    }
    
    initializeElements() {
        this.elements = {
            table: document.getElementById('txnTable'),
            tableBody: document.getElementById('txnTableBody'),
            totalTxns: document.getElementById('totalTxns'),
            highRiskCount: document.getElementById('highRiskCount'),
            mediumRiskCount: document.getElementById('mediumRiskCount'),
            lowRiskCount: document.getElementById('lowRiskCount'),
            pauseBtn: document.getElementById('pauseStream'),
            resumeBtn: document.getElementById('resumeStream'),
            streamStatus: document.getElementById('streamStatus'),
            connectionStatus: document.getElementById('connectionStatus'),
            downloadBtn: document.getElementById('downloadCsv'),
            clearBtn: document.getElementById('clearTable'),
            loadingOverlay: document.getElementById('loadingOverlay')
        };
    }
    
    setupEventListeners() {
        this.elements.pauseBtn.addEventListener('click', () => this.pauseStream());
        this.elements.resumeBtn.addEventListener('click', () => this.resumeStream());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadCSV());
        this.elements.clearBtn.addEventListener('click', () => this.clearTable());
        
        // Filter functionality (basic implementation)
        document.getElementById('applyFilters').addEventListener('click', () => this.applyFilters());
        document.getElementById('clearFilters').addEventListener('click', () => this.clearFilters());
    }
    
    async fetchStream() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/stream`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const transactions = await response.json();
            
            this.updateConnectionStatus(true);
            return transactions;
        } catch (error) {
            console.error('Error fetching stream:', error);
            this.updateConnectionStatus(false);
            return [];
        }
    }
    
    async scoreTransaction(transaction) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/score`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(transaction)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error scoring transaction:', error);
            return {
                risk_score: 0,
                risk_level: 'Low',
                is_anomaly: false
            };
        }
    }
    
    async processTransactions() {
        if (this.isPaused) return;
        
        this.showLoading(true);
        
        try {
            const transactions = await this.fetchStream();
            
            for (const transaction of transactions) {
                const score = await this.scoreTransaction(transaction);
                
                const enrichedTransaction = {
                    ...transaction,
                    risk_score: score.risk_score,
                    risk_level: score.risk_level,
                    is_anomaly: score.is_anomaly
                };
                
                this.addTransactionToTable(enrichedTransaction);
                this.transactions.push(enrichedTransaction);
                
                // Keep only latest 100 transactions
                if (this.transactions.length > this.maxRows) {
                    this.transactions.shift();
                    this.removeOldestRow();
                }
            }
            
            this.updateSummaryCards();
        } catch (error) {
            console.error('Error processing transactions:', error);
        } finally {
            this.showLoading(false);
        }
    }
    
    addTransactionToTable(transaction) {
        const row = document.createElement('tr');
        row.className = transaction.is_anomaly ? 'anomaly-row' : '';
        
        const riskClass = transaction.risk_level.toLowerCase();
        const actionText = this.getActionText(transaction.risk_level);
        const actionClass = this.getActionClass(transaction.risk_level);
        
        row.innerHTML = `
    <td>${transaction.time || this.formatTime(transaction.step)}</td>
    <td>${transaction.customer}</td>
    <td class="amount">$${parseFloat(transaction.amount).toFixed(2)}</td>
    <td>${transaction.merchant}</td>
    <td>${transaction.category}</td>
    <td class="score">${transaction.risk_score.toFixed(3)}</td>
    <td><span class="risk-badge risk-${riskClass}">${transaction.risk_level}</span></td>
    <td><button class="action-btn action-${actionClass}">${actionText}</button></td>
`;

// ✅ attach the click handler **after** the HTML is inserted
const actionBtn = row.querySelector('button');
actionBtn.addEventListener('click', () => {
    this.handleAction(transaction.customer, transaction.risk_level);
});

        
        this.elements.tableBody.insertBefore(row, this.elements.tableBody.firstChild);
    }
    
    removeOldestRow() {
        const rows = this.elements.tableBody.querySelectorAll('tr');
        if (rows.length > this.maxRows) {
            rows[rows.length - 1].remove();
        }
    }
    
    formatTime(step) {
        const baseDate = new Date('2011-01-01');
        const transactionDate = new Date(baseDate.getTime() + step * 60000);
        return transactionDate.toLocaleString();
    }
    
    getActionText(riskLevel) {
        switch (riskLevel) {
            case 'High': return 'Investigate';
            case 'Medium': return 'Review';
            case 'Low': return 'Approve';
            default: return 'Review';
        }
    }
    
    getActionClass(riskLevel) {
        switch (riskLevel) {
            case 'High': return 'investigate';
            case 'Medium': return 'review';
            case 'Low': return 'approve';
            default: return 'review';
        }
    }
    
    handleAction(customerId, riskLevel) {
        const message = `Action triggered for customer ${customerId} with ${riskLevel} risk level`;
        alert(message);
        console.log(message);
    }
    
    updateSummaryCards() {
        const total = this.transactions.length;
        const highRisk = this.transactions.filter(t => t.risk_level === 'High').length;
        const mediumRisk = this.transactions.filter(t => t.risk_level === 'Medium').length;
        const lowRisk = this.transactions.filter(t => t.risk_level === 'Low').length;
        
        this.elements.totalTxns.textContent = total;
        this.elements.highRiskCount.textContent = highRisk;
        this.elements.mediumRiskCount.textContent = mediumRisk;
        this.elements.lowRiskCount.textContent = lowRisk;
    }
    
    updateConnectionStatus(isConnected) {
        this.elements.connectionStatus.className = `status-dot ${isConnected ? 'online' : 'offline'}`;
    }
    
    startStreaming() {
        if (this.isStreaming) return;
        
        this.isStreaming = true;
        this.isPaused = false;
        this.elements.streamStatus.textContent = 'Streaming...';
        
        // Initial load
        this.processTransactions();
        
        // Set up interval for continuous streaming
        this.streamInterval = setInterval(() => {
            this.processTransactions();
        }, 5000); // Every 5 seconds
    }
    
    pauseStream() {
        this.isPaused = true;
        this.elements.pauseBtn.style.display = 'none';
        this.elements.resumeBtn.style.display = 'inline-block';
        this.elements.streamStatus.textContent = 'Paused';
    }
    
    resumeStream() {
        this.isPaused = false;
        this.elements.pauseBtn.style.display = 'inline-block';
        this.elements.resumeBtn.style.display = 'none';
        this.elements.streamStatus.textContent = 'Streaming...';
    }
    
    stopStreaming() {
        this.isStreaming = false;
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }
        this.elements.streamStatus.textContent = 'Stopped';
    }
    
    clearTable() {
        this.elements.tableBody.innerHTML = '';
        this.transactions = [];
        this.updateSummaryCards();
    }
    
    showLoading(show) {
        this.elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }
    
    downloadCSV() {
        if (this.transactions.length === 0) {
            alert('No transactions to download');
            return;
        }
        
        const headers = ['Time', 'Customer ID', 'Amount', 'Merchant', 'Category', 'Risk Score', 'Risk Level', 'Is Anomaly'];
        const csvContent = [
            headers.join(','),
            ...this.transactions.map(t => [
                t.time || this.formatTime(t.step),
                t.customer,
                t.amount,
                t.merchant,
                t.category,
                t.risk_score.toFixed(3),
                t.risk_level,
                t.is_anomaly
            ].join(','))
        ].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', `transactions_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    applyFilters() {
        const customerFilter = document.getElementById('customerFilter').value.toLowerCase();
        const riskFilter = document.getElementById('riskFilter').value;
        
        const rows = this.elements.tableBody.querySelectorAll('tr');
        
        rows.forEach(row => {
            const customer = row.cells[1].textContent.toLowerCase();
            const riskLevel = row.cells[6].textContent.trim();
            
            const matchesCustomer = !customerFilter || customer.includes(customerFilter);
            const matchesRisk = !riskFilter || riskLevel === riskFilter;
            
            row.style.display = matchesCustomer && matchesRisk ? '' : 'none';
        });
    }
    
    clearFilters() {
        document.getElementById('customerFilter').value = '';
        document.getElementById('riskFilter').value = '';
        document.getElementById('dateFrom').value = '';
        document.getElementById('dateTo').value = '';
        
        const rows = this.elements.tableBody.querySelectorAll('tr');
        rows.forEach(row => {
            row.style.display = '';
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.transactionMonitor = new TransactionMonitor();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - streaming continues in background');
    } else {
        console.log('Page visible - streaming active');
    }
});

// Handle window beforeunload
window.addEventListener('beforeunload', () => {
    if (window.transactionMonitor) {
        window.transactionMonitor.stopStreaming();
    }
});