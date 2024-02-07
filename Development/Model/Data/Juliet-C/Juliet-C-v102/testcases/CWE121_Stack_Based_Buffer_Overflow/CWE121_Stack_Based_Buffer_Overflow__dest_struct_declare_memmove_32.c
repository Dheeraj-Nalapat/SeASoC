/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__dest_struct_declare_memmove_32.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__dest.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Set data pointer to the bad buffer
 * GoodSource: Set data pointer to the good buffer
 * Sink: memmove
 *    BadSink : Copy twoints array to data using memmove
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__dest_struct_declare_memmove_32_bad()
{
    twoints * data;
    twoints * *data_ptr1 = &data;
    twoints * *data_ptr2 = &data;
    twoints data_badbuf[50];
    twoints data_goodbuf[100];
    {
        twoints * data = *data_ptr1;
        /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
         * buffer in various memory copying functions using a "large" source buffer. */
        data = data_badbuf;
        *data_ptr1 = data;
    }
    {
        twoints * data = *data_ptr2;
        {
            twoints src[100];
            {
                size_t i;
                /* Initialize array */
                for (i = 0; i < 100; i++)
                {
                    src[i].a = 0;
                    src[i].b = 0;
                }
            }
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            memmove(data, src, 100*sizeof(twoints));
            printStructLine(&data[0]);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    twoints * data;
    twoints * *data_ptr1 = &data;
    twoints * *data_ptr2 = &data;
    twoints data_badbuf[50];
    twoints data_goodbuf[100];
    {
        twoints * data = *data_ptr1;
        /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
        data = data_goodbuf;
        *data_ptr1 = data;
    }
    {
        twoints * data = *data_ptr2;
        {
            twoints src[100];
            {
                size_t i;
                /* Initialize array */
                for (i = 0; i < 100; i++)
                {
                    src[i].a = 0;
                    src[i].b = 0;
                }
            }
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            memmove(data, src, 100*sizeof(twoints));
            printStructLine(&data[0]);
        }
    }
}

void CWE121_Stack_Based_Buffer_Overflow__dest_struct_declare_memmove_32_good()
{
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
    CWE121_Stack_Based_Buffer_Overflow__dest_struct_declare_memmove_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__dest_struct_declare_memmove_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
