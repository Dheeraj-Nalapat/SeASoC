/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__unsigned_int_rand_multiply_53d.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-53d.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: rand Set data to result of rand()
 * GoodSource: Small, non-zero
 * Sinks: multiply
 *    GoodSink: Ensure there is no overflow before performing the multiplication
 *    BadSink : Multiply data by 2
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE190_Integer_Overflow__unsigned_int_rand_multiply_53d_bad_sink(unsigned int data)
{
    {
        /* POTENTIAL FLAW: Multiplying data by 2 could cause an overflow */
        unsigned int result = data * 2;
        printUnsignedLine(result);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE190_Integer_Overflow__unsigned_int_rand_multiply_53d_goodG2B_sink(unsigned int data)
{
    {
        /* POTENTIAL FLAW: Multiplying data by 2 could cause an overflow */
        unsigned int result = data * 2;
        printUnsignedLine(result);
    }
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE190_Integer_Overflow__unsigned_int_rand_multiply_53d_goodB2G_sink(unsigned int data)
{
    {
        unsigned int result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (UINT_MAX/2))
        {
            result = data * 2;
            printUnsignedLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

#endif /* OMITGOOD */
