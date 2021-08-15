import React from "react"
import Layout from "../components/layout"
import {Link} from "gatsby"

var projects = require('../content/projects.json');
console.log(projects);

export default function Projects() {
    return (
        <Layout>
            <h1>Projects</h1>
            {
                projects.map(el => 
                    <div>
                        <Link to={el.url} style={{ color:"inherit",textDecoration:"none",textDecorationLine:"none"}}>
                            <h4>{el.date}. {el.name}<span style={{color:"#777"}}> {el.description}</span></h4>
                        </Link>
                    </div>
                )
            }
        </Layout>
    )
}
