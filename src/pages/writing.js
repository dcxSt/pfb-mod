import React from "react"
import { graphql } from "gatsby"
import Sidenav from "../components/sidebar"
import { Link } from "gatsby"

export default function Writing({ data }) {
    console.log(data);
    return (
        <div style={{ color:`black`,margin:`3rem auto`}}>
            <Sidenav />
            <div class="main">
                <h1>Writing</h1>
                {data.allMarkdownRemark.edges.map(({ node }) => {
                    if (node.fileAbsolutePath.includes('writing'))
                        return (
                            <div key={node.id}>
                                <Link to={node.fields.slug} style={{ color:"inherit",textDectoration:"none",textDecorationLine:"none"}}><h4>{node.frontmatter.title} <span style={{color:"#777"}}> -{node.timeToRead} minuet read</span></h4></Link>
                                {node.excerpt}
                            </div>
                        )
                        // seems to order by date by default!
                })}
            </div>
        </div>
    );
}

export const query = graphql`
    query {
        allMarkdownRemark (sort: { fields: [frontmatter___date], order:DESC}) {
            edges {
            node {
                id
                excerpt
                frontmatter {
                    title
                    date
                }
                fileAbsolutePath
                timeToRead
                fields {
                    slug
                }
            }
            }
        }
    }
`