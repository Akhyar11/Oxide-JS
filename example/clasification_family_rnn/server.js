import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const LOG_DIR = path.join(__dirname, 'log');

const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/dashboard') {
        fs.readFile(path.join(LOG_DIR, 'stability_report.html'), (err, data) => {
            if (err) {
                res.writeHead(404);
                res.end("File stability_report.html not found in log directory.");
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else if (req.url === '/api/data') {
        fs.readdir(LOG_DIR, (err, files) => {
            if (err) {
                res.writeHead(500);
                res.end("Failed to read log directory");
                return;
            }

            const jsonFiles = files
                .filter(f => f.startsWith('stability_experiment_') && f.endsWith('.json'))
                .sort()
                .reverse();

            if (jsonFiles.length === 0) {
                res.writeHead(404);
                res.end(JSON.stringify({ error: "No experiment data found" }));
                return;
            }

            const latestFile = path.join(LOG_DIR, jsonFiles[0]);
            fs.readFile(latestFile, (readErr, data) => {
                if (readErr) {
                    res.writeHead(500);
                    res.end("Failed to read JSON file");
                    return;
                }
                res.writeHead(200, {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                });
                res.end(data);
            });
        });
    } else if (req.url === '/api/eval_data') {
        fs.readdir(LOG_DIR, (err, files) => {
            if (err) {
                res.writeHead(500);
                res.end("Failed to read log directory");
                return;
            }

            const jsonFiles = files
                .filter(f => f.startsWith('checkpoint_evaluation_') && f.endsWith('.json'))
                .sort()
                .reverse();

            if (jsonFiles.length === 0) {
                res.writeHead(404);
                res.end(JSON.stringify({ error: "No evaluation data found" }));
                return;
            }

            const latestFile = path.join(LOG_DIR, jsonFiles[0]);
            fs.readFile(latestFile, (readErr, data) => {
                if (readErr) {
                    res.writeHead(500);
                    res.end("Failed to read JSON file");
                    return;
                }
                res.writeHead(200, {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                });
                res.end(data);
            });
        });
    } else {
        res.writeHead(404);
        res.end("Not Found");
    }
});

server.listen(PORT, () => {
    console.log(`\n🚀 Dashboard Server running at: http://localhost:${PORT}`);
    console.log(`📁 Connecting data from: ${LOG_DIR}\n`);
});
