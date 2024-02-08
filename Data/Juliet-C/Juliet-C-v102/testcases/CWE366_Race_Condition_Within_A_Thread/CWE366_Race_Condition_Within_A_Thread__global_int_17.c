/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE366_Race_Condition_Within_A_Thread__global_int_17.c
Label Definition File: CWE366_Race_Condition_Within_A_Thread.label.xml
Template File: point-flaw-17.tmpl.c
*/
/*
 * @description
 * CWE: 366 Race Condition Within a Thread
 * Sinks: global_int
 *    GoodSink: Acquire a lock before conducting operations
 *    BadSink : Do not acquire a lock before conducting operations
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#include "std_thread.h"

#define N_ITERS 1000000

static int g_bad = 0;
static int g_good = 0;
static std_thread_lock g_good_lock = NULL;

static void helper_bad(void *args)
{
    int i;
    /* FLAW: incrementing an integer is not guaranteed to occur atomically;
     * therefore this operation may not function as intended in multi-threaded
     * programs
     */
    /* I'm going to risk going out on a limb here and making this slightly
     * more complicated to illustrate the point: doing this in a loop a million
     * times makes it much more "obvious" that something wrong might happen
     * (you can even see it in action when you run the program)
     */
    for (i = 0; i < N_ITERS; i++)
    {
        g_bad = g_bad + 1;
    }
}

static void helper_good(void *args)
{
    int i;
    /* FIX: acquire a lock before conducting operations that need to occur
     * atomically, and release afterwards
     */
    std_thread_lock_acquire(g_good_lock);
    for (i = 0; i < N_ITERS; i++)
    {
        g_good = g_good + 1;
    }
    std_thread_lock_release(g_good_lock);
}

#ifndef OMITBAD

void CWE366_Race_Condition_Within_A_Thread__global_int_17_bad()
{
    int j,k;
    for(j = 0; j < 1; j++)
    {
        {
            std_thread thread_a = NULL;
            std_thread thread_b = NULL;
            g_bad = 0;
            if (!std_thread_create(helper_bad, NULL, &thread_a))
            {
                thread_a = NULL;
            }
            if (!std_thread_create(helper_bad, NULL, &thread_b))
            {
                thread_b = NULL;
            }
            if (thread_a && std_thread_join(thread_a)) std_thread_destroy(thread_a);
            if (thread_b && std_thread_join(thread_b)) std_thread_destroy(thread_b);
            printIntLine(g_bad);
        }
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            std_thread thread_a = NULL;
            std_thread thread_b = NULL;
            if (!std_thread_lock_create(&g_good_lock))
            {
                return;
            }
            if (!std_thread_create(helper_good, NULL, &thread_a))
            {
                thread_a = NULL;
            }
            if (!std_thread_create(helper_good, NULL, &thread_b))
            {
                thread_b = NULL;
            }
            if (thread_a && std_thread_join(thread_a)) std_thread_destroy(thread_a);
            if (thread_b && std_thread_join(thread_b)) std_thread_destroy(thread_b);
            std_thread_lock_destroy(g_good_lock);
            printIntLine(g_good);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the conditions on the for statements */
static void good1()
{
    int j,k;
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            std_thread thread_a = NULL;
            std_thread thread_b = NULL;
            g_bad = 0;
            if (!std_thread_create(helper_bad, NULL, &thread_a))
            {
                thread_a = NULL;
            }
            if (!std_thread_create(helper_bad, NULL, &thread_b))
            {
                thread_b = NULL;
            }
            if (thread_a && std_thread_join(thread_a)) std_thread_destroy(thread_a);
            if (thread_b && std_thread_join(thread_b)) std_thread_destroy(thread_b);
            printIntLine(g_bad);
        }
    }
    for(k = 0; k < 1; k++)
    {
        {
            std_thread thread_a = NULL;
            std_thread thread_b = NULL;
            if (!std_thread_lock_create(&g_good_lock))
            {
                return;
            }
            if (!std_thread_create(helper_good, NULL, &thread_a))
            {
                thread_a = NULL;
            }
            if (!std_thread_create(helper_good, NULL, &thread_b))
            {
                thread_b = NULL;
            }
            if (thread_a && std_thread_join(thread_a)) std_thread_destroy(thread_a);
            if (thread_b && std_thread_join(thread_b)) std_thread_destroy(thread_b);
            std_thread_lock_destroy(g_good_lock);
            printIntLine(g_good);
        }
    }
}

void CWE366_Race_Condition_Within_A_Thread__global_int_17_good()
{
    good1();
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
    CWE366_Race_Condition_Within_A_Thread__global_int_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE366_Race_Condition_Within_A_Thread__global_int_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
