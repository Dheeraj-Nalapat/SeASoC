/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__int_fscanf_add_34.c
Label Definition File: CWE190_Integer_Overflow__int.label.xml
Template File: sources-sinks-34.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Small, non-zero
 * Sinks: add
 *    GoodSink: Ensure there is no overflow before performing the addition
 *    BadSink : Add 1 to data
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

typedef union
{
    int a;
    int b;
} CWE190_Integer_Overflow__int_fscanf_add_34_union_type;

#ifndef OMITBAD

void CWE190_Integer_Overflow__int_fscanf_add_34_bad()
{
    int data;
    CWE190_Integer_Overflow__int_fscanf_add_34_union_type my_union;
    /* Initialize data */
    data = -1;
    fscanf (stdin, "%d", &data);
    my_union.a = data;
    {
        int data = my_union.b;
        {
            /* POTENTIAL FLAW: Adding 1 to data could cause an integer overflow */
            int result = data + 1;
            printIntLine(result);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    int data;
    CWE190_Integer_Overflow__int_fscanf_add_34_union_type my_union;
    /* Initialize data */
    data = -1;
    /* FIX: Use a small, non-zero value that will not cause an integer overflow in the sinks */
    data = 5;
    my_union.a = data;
    {
        int data = my_union.b;
        {
            /* POTENTIAL FLAW: Adding 1 to data could cause an integer overflow */
            int result = data + 1;
            printIntLine(result);
        }
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    int data;
    CWE190_Integer_Overflow__int_fscanf_add_34_union_type my_union;
    /* Initialize data */
    data = -1;
    fscanf (stdin, "%d", &data);
    my_union.a = data;
    {
        int data = my_union.b;
        {
            int result = -1;
            /* FIX: Add a check to prevent an overflow from occurring */
            if (data < INT_MAX)
            {
                result = data + 1;
                printIntLine(result);
            }
            else
            {
                printLine("Input value is too large to perform arithmetic safely.");
            }
        }
    }
}

void CWE190_Integer_Overflow__int_fscanf_add_34_good()
{
    goodG2B();
    goodB2G();
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
    CWE190_Integer_Overflow__int_fscanf_add_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__int_fscanf_add_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
