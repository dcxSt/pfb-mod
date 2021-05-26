import React from "react"
import Layout from "../components/layout"
import "../index.css"

var quotes = require('../content/quotes.json');

export default function Quotes() {
    return (
        <Layout>
            <h1>Nuggets of Gold 2</h1>
            {
                quotes.map(el => <div>
                    <pquote>
                        {el.content}
                        <br/><br/>
                        <b>{el.source}</b>
                    </pquote>
                </div>)
            }
        </Layout>
    );
}