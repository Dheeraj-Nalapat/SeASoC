/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__unsigned_int_fixed_add_34.c
Label Definition File: CWE190_Integer_Overflow.label.xml
Template File: sources-sinks-34.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: fixed Fixed value
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
    unsigned int a;
    unsigned int b;
} CWE190_Integer_Overflow__unsigned_int_fixed_add_34_union_type;

#ifndef OMITBAD

void CWE190_Integer_Overflow__unsigned_int_fixed_add_34_bad()
{
    unsigned int data;
    CWE190_Integer_Overflow__unsigned_int_fixed_add_34_union_type my_union;
    data = 0;
    /* FLAW: Use the maximum size of the data type */
    data = UINT_MAX;
    my_union.a = data;
    {
        unsigned int data = my_union.b;
        {
            /* POTENTIAL FLAW: Adding 1 to data could cause an overflow */
            unsigned int result = data + 1;
            printUnsignedLine(result);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    unsigned int data;
    CWE190_Integer_Overflow__unsigned_int_fixed_add_34_union_type my_union;
    data = 0;
    /* FIX: Use a small, non-zero value that will not cause an overflow in the sinks */
    data = 5;
    my_union.a = data;
    {
        unsigned int data = my_union.b;
        {
            /* POTENTIAL FLAW: Adding 1 to data could cause an overflow */
            unsigned int result = data + 1;
            printUnsignedLine(result);
        }
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    unsigned int data;
    CWE190_Integer_Overflow__unsigned_int_fixed_add_34_union_type my_union;
    data = 0;
    /* FLAW: Use the maximum size of the data type */
    data = UINT_MAX;
    my_union.a = data;
    {
        unsigned int data = my_union.b;
        {
            unsigned int result = -1;
            /* FIX: Add a check to prevent an overflow from occurring */
            if (data < UINT_MAX)
            {
                result = data + 1;
                printUnsignedLine(result);
            }
            else
            {
                printLine("Input value is too large to perform arithmetic safely.");
            }
        }
    }
}

void CWE190_Integer_Overflow__unsigned_int_fixed_add_34_good()
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
    CWE190_Integer_Overflow__unsigned_int_fixed_add_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__unsigned_int_fixed_add_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
