/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE191_Integer_Underflow__unsigned_char_rand_61b.c
Label Definition File: CWE191_Integer_Underflow.label.xml
Template File: sources-sinks-61b.tmpl.c
*/
/*
 * @description
 * CWE: 191 Integer Underflow
 * BadSource: rand Set data to result of rand()
 * GoodSource: Small, non-zero
 * Sinks:
 *    GoodSink: Ensure there is no underflow before performing the subtraction
 *    BadSink : Subtract 1 from data
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

unsigned char CWE191_Integer_Underflow__unsigned_char_rand_61b_bad_source(unsigned char data)
{
    /* POTENTIAL FLAW: Use a random value */
    data = (unsigned char)rand();
    return data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
unsigned char CWE191_Integer_Underflow__unsigned_char_rand_61b_goodG2B_source(unsigned char data)
{
    /* FIX: Use a small value greater than the min value for this data type */
    data = 5;
    return data;
}

/* goodB2G() uses the BadSource with the GoodSink */
unsigned char CWE191_Integer_Underflow__unsigned_char_rand_61b_goodB2G_source(unsigned char data)
{
    /* POTENTIAL FLAW: Use a random value */
    data = (unsigned char)rand();
    return data;
}

#endif /* OMITGOOD */
