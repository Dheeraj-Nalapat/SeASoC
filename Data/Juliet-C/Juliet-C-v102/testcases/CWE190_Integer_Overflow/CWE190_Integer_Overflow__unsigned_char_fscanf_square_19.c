/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__unsigned_char_fscanf_square_19.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-19.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Small, non-zero
 * Sinks: square
 *    GoodSink: Ensure there is no overflow before performing the squaring operation
 *    BadSink : Square data
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#include <math.h>

#ifndef OMITBAD

void CWE190_Integer_Overflow__unsigned_char_fscanf_square_19_bad()
{
    unsigned char data;
    data = ' ';
    /* POTENTIAL FLAW: Use a value input from the console */
    fscanf (stdin, "%hc", &data);
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        unsigned char result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (unsigned char)sqrt((unsigned char)UCHAR_MAX))
        {
            result = data * data;
            printHexUnsignedCharLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by switching statements around return */
static void goodB2G()
{
    unsigned char data;
    data = ' ';
    /* POTENTIAL FLAW: Use a value input from the console */
    fscanf (stdin, "%hc", &data);
    {
        unsigned char result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data <= (unsigned char)sqrt((unsigned char)UCHAR_MAX))
        {
            result = data * data;
            printHexUnsignedCharLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
}

/* goodG2B() - use goodsource and badsink by moving BadSource and BadSink to after return */
static void goodG2B()
{
    unsigned char data;
    data = ' ';
    /* FIX: Use a small, non-zero value that will not cause an overflow in the sinks */
    data = 5;
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* POTENTIAL FLAW: Use a value input from the console */
    fscanf (stdin, "%hc", &data);
    {
        /* POTENTIAL FLAW: Squaring data could cause an overflow */
        unsigned char result = data * data;
        printHexUnsignedCharLine(result);
    }
}

void CWE190_Integer_Overflow__unsigned_char_fscanf_square_19_good()
{
    goodB2G();
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE190_Integer_Overflow__unsigned_char_fscanf_square_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__unsigned_char_fscanf_square_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
