/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__unsigned_int_rand_square_64b.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-64b.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: rand Set data to result of rand()
 * GoodSource: Small, non-zero
 * Sinks: square
 *    GoodSink: Ensure there is no overflow before performing the squaring operation
 *    BadSink : Square data
 * Flow Variant: 64 Data flow: void pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <math.h>

#ifndef OMITBAD

void CWE190_Integer_Overflow__unsigned_int_rand_square_64b_bad_sink(void * void_data_ptr)
{
    /* cast void pointer to a pointer of the appropriate type */
    unsigned int * data_ptr = (unsigned int *)void_data_ptr;
    /* dereference data_ptr into data */
    unsigned int data = (*data_ptr);
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned int result = data * data;
        printUnsignedLine(result);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE190_Integer_Overflow__unsigned_int_rand_square_64b_goodG2B_sink(void * void_data_ptr)
{
    /* cast void pointer to a pointer of the appropriate type */
    unsigned int * data_ptr = (unsigned int *)void_data_ptr;
    /* dereference data_ptr into data */
    unsigned int data = (*data_ptr);
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned int result = data * data;
        printUnsignedLine(result);
    }
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE190_Integer_Overflow__unsigned_int_rand_square_64b_goodB2G_sink(void * void_data_ptr)
{
    /* cast void pointer to a pointer of the appropriate type */
    unsigned int * data_ptr = (unsigned int *)void_data_ptr;
    /* dereference data_ptr into data */
    unsigned int data = (*data_ptr);
    {
        unsigned int result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (unsigned int)sqrt((unsigned int)UINT_MAX))
        {
            result = data * data;
            printUnsignedLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

#endif /* OMITGOOD */
