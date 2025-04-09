	.file	"HARSHMISHRA-521_SYSTEM-SOFTWARE-AND-COMPILERS-LAB_7_flatten.c"
	.text
	.globl	_TIG_IZ_ZVy6_argv
	.bss
	.align 8
	.type	_TIG_IZ_ZVy6_argv, @object
	.size	_TIG_IZ_ZVy6_argv, 8
_TIG_IZ_ZVy6_argv:
	.zero	8
	.globl	_TIG_IZ_ZVy6_argc
	.align 4
	.type	_TIG_IZ_ZVy6_argc, @object
	.size	_TIG_IZ_ZVy6_argc, 4
_TIG_IZ_ZVy6_argc:
	.zero	4
	.globl	_TIG_IZ_ZVy6_envp
	.align 8
	.type	_TIG_IZ_ZVy6_envp, @object
	.size	_TIG_IZ_ZVy6_envp, 8
_TIG_IZ_ZVy6_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\nRound Robin scheduling algorithm\nEnter the number of processes: "
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"\nEnter the burst time for sequences: "
	.align 8
.LC3:
	.string	"\n\nShortest Remaining Time First (SRTF)"
.LC4:
	.string	"\nEnter the time quantum: "
	.align 8
.LC5:
	.string	"\n1. Round Robin\n2. SRTF\n3. Exit\nEnter the choice: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_ZVy6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ZVy6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ZVy6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 134 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ZVy6--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_ZVy6_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_ZVy6_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_ZVy6_envp(%rip)
	nop
	movq	$15, -104(%rbp)
.L26:
	cmpq	$17, -104(%rbp)
	ja	.L28
	movq	-104(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L28-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L9:
	movq	$2, -104(%rbp)
	jmp	.L18
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-120(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -108(%rbp)
	movq	$17, -104(%rbp)
	jmp	.L18
.L16:
	movq	$2, -104(%rbp)
	jmp	.L18
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	srtf
	movq	$2, -104(%rbp)
	jmp	.L18
.L14:
	movl	-112(%rbp), %eax
	cmpl	$3, %eax
	je	.L19
	cmpl	$3, %eax
	jg	.L20
	cmpl	$1, %eax
	je	.L21
	cmpl	$2, %eax
	je	.L22
	jmp	.L20
.L19:
	movq	$6, -104(%rbp)
	jmp	.L23
.L22:
	movq	$11, -104(%rbp)
	jmp	.L23
.L21:
	movq	$12, -104(%rbp)
	jmp	.L23
.L20:
	movq	$3, -104(%rbp)
	nop
.L23:
	jmp	.L18
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-116(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-116(%rbp), %esi
	movl	-120(%rbp), %eax
	leaq	-96(%rbp), %rcx
	leaq	-48(%rbp), %rdx
	movl	%eax, %edi
	call	roundrobin
	movq	$2, -104(%rbp)
	jmp	.L18
.L7:
	movl	-120(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jge	.L24
	movq	$10, -104(%rbp)
	jmp	.L18
.L24:
	movq	$13, -104(%rbp)
	jmp	.L18
.L15:
	movl	$0, %edi
	call	exit@PLT
.L13:
	leaq	-96(%rbp), %rdx
	movl	-108(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-108(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %edx
	movl	-108(%rbp), %eax
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	addl	$1, -108(%rbp)
	movq	$17, -104(%rbp)
	jmp	.L18
.L17:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -104(%rbp)
	jmp	.L18
.L28:
	nop
.L18:
	jmp	.L26
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC6:
	.string	"Process no\t|Burst time\t|Waiting time\t|Turnaround time"
	.align 8
.LC8:
	.string	"\nAverage waiting time is %f\nAverage turnaround time is %f\n"
.LC9:
	.string	"%d\t\t\t%d\t\t\t%d\t\t\t%d\t\t\n"
	.text
	.globl	roundrobin
	.type	roundrobin, @function
roundrobin:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movl	%esi, -152(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rcx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$10, -104(%rbp)
.L70:
	cmpq	$36, -104(%rbp)
	ja	.L73
	movq	-104(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L53-.L32
	.long	.L73-.L32
	.long	.L52-.L32
	.long	.L51-.L32
	.long	.L50-.L32
	.long	.L73-.L32
	.long	.L74-.L32
	.long	.L48-.L32
	.long	.L47-.L32
	.long	.L73-.L32
	.long	.L46-.L32
	.long	.L73-.L32
	.long	.L45-.L32
	.long	.L44-.L32
	.long	.L43-.L32
	.long	.L42-.L32
	.long	.L73-.L32
	.long	.L41-.L32
	.long	.L73-.L32
	.long	.L40-.L32
	.long	.L73-.L32
	.long	.L73-.L32
	.long	.L73-.L32
	.long	.L39-.L32
	.long	.L38-.L32
	.long	.L37-.L32
	.long	.L73-.L32
	.long	.L36-.L32
	.long	.L73-.L32
	.long	.L73-.L32
	.long	.L73-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L33-.L32
	.long	.L73-.L32
	.long	.L73-.L32
	.long	.L31-.L32
	.text
.L37:
	movl	$0, -124(%rbp)
	movl	$0, -140(%rbp)
	movq	$2, -104(%rbp)
	jmp	.L54
.L50:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L55
	movq	$8, -104(%rbp)
	jmp	.L54
.L55:
	movq	$24, -104(%rbp)
	jmp	.L54
.L43:
	movl	-124(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L57
	movq	$7, -104(%rbp)
	jmp	.L54
.L57:
	movq	$31, -104(%rbp)
	jmp	.L54
.L42:
	movl	-148(%rbp), %eax
	cmpl	-140(%rbp), %eax
	jne	.L59
	movq	$0, -104(%rbp)
	jmp	.L54
.L59:
	movq	$25, -104(%rbp)
	jmp	.L54
.L35:
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-136(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2ssl	-148(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -116(%rbp)
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-120(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2ssl	-148(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -112(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -124(%rbp)
	movq	$32, -104(%rbp)
	jmp	.L54
.L45:
	movl	$0, -108(%rbp)
	movl	$0, -140(%rbp)
	movl	$0, -136(%rbp)
	movl	$0, -128(%rbp)
	movl	$0, -120(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -116(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -112(%rbp)
	movq	$25, -104(%rbp)
	jmp	.L54
.L47:
	addl	$1, -140(%rbp)
	movq	$36, -104(%rbp)
	jmp	.L54
.L39:
	movl	-132(%rbp), %eax
	addl	%eax, -128(%rbp)
	movl	-124(%rbp), %eax
	cltq
	movl	-128(%rbp), %edx
	movl	%edx, -96(%rbp,%rax,4)
	movq	$36, -104(%rbp)
	jmp	.L54
.L51:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	js	.L61
	movq	$13, -104(%rbp)
	jmp	.L54
.L61:
	movq	$23, -104(%rbp)
	jmp	.L54
.L38:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -152(%rbp)
	jge	.L63
	movq	$33, -104(%rbp)
	jmp	.L54
.L63:
	movq	$3, -104(%rbp)
	jmp	.L54
.L31:
	addl	$1, -124(%rbp)
	movq	$2, -104(%rbp)
	jmp	.L54
.L44:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$23, -104(%rbp)
	jmp	.L54
.L40:
	pxor	%xmm0, %xmm0
	cvtss2sd	-112(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	-116(%rbp), %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$6, -104(%rbp)
	jmp	.L54
.L34:
	movl	-124(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L65
	movq	$27, -104(%rbp)
	jmp	.L54
.L65:
	movq	$19, -104(%rbp)
	jmp	.L54
.L41:
	movl	-152(%rbp), %eax
	movl	%eax, -132(%rbp)
	movq	$4, -104(%rbp)
	jmp	.L54
.L36:
	movl	-124(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %edi
	movl	-124(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-168(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	-124(%rbp), %ecx
	leal	1(%rcx), %esi
	movl	%edi, %r8d
	movl	%edx, %ecx
	movl	%eax, %edx
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -124(%rbp)
	movq	$32, -104(%rbp)
	jmp	.L54
.L33:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	-124(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-160(%rbp), %rdx
	addq	%rcx, %rdx
	subl	-152(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$23, -104(%rbp)
	jmp	.L54
.L46:
	movq	$12, -104(%rbp)
	jmp	.L54
.L53:
	movl	$0, -124(%rbp)
	movq	$14, -104(%rbp)
	jmp	.L54
.L48:
	movl	-124(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %eax
	movl	-124(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-168(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	movl	-124(%rbp), %eax
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	movl	-124(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	addl	%eax, -136(%rbp)
	movl	-124(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %eax
	addl	%eax, -120(%rbp)
	addl	$1, -124(%rbp)
	movq	$14, -104(%rbp)
	jmp	.L54
.L52:
	movl	-124(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L68
	movq	$17, -104(%rbp)
	jmp	.L54
.L68:
	movq	$15, -104(%rbp)
	jmp	.L54
.L73:
	nop
.L54:
	jmp	.L70
.L74:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L72
	call	__stack_chk_fail@PLT
.L72:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	roundrobin, .-roundrobin
	.section	.rodata
	.align 8
.LC10:
	.string	"Enter arrival time for p[%d]: "
	.align 8
.LC11:
	.string	"Enter the burst time for p[%d]: "
	.align 8
.LC12:
	.string	"\nEnter the number of processes: "
.LC13:
	.string	"p[%d]\t\t%d\t\t%d\n"
	.align 8
.LC14:
	.string	"\nProcess\t\tWaiting time\tTurnaround time"
.LC15:
	.string	"Average waiting time: %f\n"
.LC16:
	.string	"Average turnaround time: %f\n"
	.text
	.globl	srtf
	.type	srtf, @function
srtf:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$9, -152(%rbp)
.L113:
	cmpq	$29, -152(%rbp)
	ja	.L99
	movq	-152(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L96-.L78
	.long	.L95-.L78
	.long	.L99-.L78
	.long	.L94-.L78
	.long	.L93-.L78
	.long	.L99-.L78
	.long	.L92-.L78
	.long	.L91-.L78
	.long	.L90-.L78
	.long	.L89-.L78
	.long	.L99-.L78
	.long	.L88-.L78
	.long	.L99-.L78
	.long	.L99-.L78
	.long	.L87-.L78
	.long	.L99-.L78
	.long	.L86-.L78
	.long	.L85-.L78
	.long	.L99-.L78
	.long	.L116-.L78
	.long	.L83-.L78
	.long	.L99-.L78
	.long	.L82-.L78
	.long	.L81-.L78
	.long	.L80-.L78
	.long	.L99-.L78
	.long	.L79-.L78
	.long	.L99-.L78
	.long	.L99-.L78
	.long	.L77-.L78
	.text
.L93:
	movl	-180(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-188(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	cmpl	%eax, %edx
	jge	.L97
	movq	$11, -152(%rbp)
	jmp	.L99
.L97:
	movq	$3, -152(%rbp)
	jmp	.L99
.L87:
	movl	-180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-144(%rbp), %rdx
	movl	-180(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-96(%rbp), %rdx
	movl	-180(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-180(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %edx
	movl	-180(%rbp), %eax
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	addl	$1, -180(%rbp)
	movq	$17, -152(%rbp)
	jmp	.L99
.L90:
	movl	-188(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	testl	%eax, %eax
	jne	.L100
	movq	$24, -152(%rbp)
	jmp	.L99
.L100:
	movq	$1, -152(%rbp)
	jmp	.L99
.L95:
	addl	$1, -184(%rbp)
	movq	$26, -152(%rbp)
	jmp	.L99
.L81:
	movl	$0, -156(%rbp)
	movl	$0, -192(%rbp)
	movl	$0, -184(%rbp)
	movl	$0, -176(%rbp)
	movl	$0, -172(%rbp)
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-196(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -180(%rbp)
	movq	$17, -152(%rbp)
	jmp	.L99
.L94:
	addl	$1, -180(%rbp)
	movq	$16, -152(%rbp)
	jmp	.L99
.L86:
	movl	-196(%rbp), %eax
	cmpl	%eax, -180(%rbp)
	jge	.L102
	movq	$6, -152(%rbp)
	jmp	.L99
.L102:
	movq	$0, -152(%rbp)
	jmp	.L99
.L80:
	addl	$1, -192(%rbp)
	movl	-184(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -160(%rbp)
	movl	-188(%rbp), %eax
	movl	%eax, -156(%rbp)
	movl	-156(%rbp), %eax
	cltq
	movl	-144(%rbp,%rax,4), %edx
	movl	-160(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %ecx
	movl	-156(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %esi
	movl	-160(%rbp), %eax
	subl	%esi, %eax
	movl	%eax, %edx
	movl	-156(%rbp), %eax
	cltq
	movl	-144(%rbp,%rax,4), %eax
	subl	%eax, %edx
	movl	-188(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-156(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %ecx
	movl	-160(%rbp), %eax
	subl	%ecx, %eax
	movl	%eax, %edx
	movl	-156(%rbp), %eax
	cltq
	movl	-144(%rbp,%rax,4), %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	addl	%eax, -176(%rbp)
	movl	-156(%rbp), %eax
	cltq
	movl	-144(%rbp,%rax,4), %edx
	movl	-160(%rbp), %eax
	subl	%edx, %eax
	addl	%eax, -172(%rbp)
	movq	$1, -152(%rbp)
	jmp	.L99
.L79:
	movl	-196(%rbp), %eax
	cmpl	%eax, -192(%rbp)
	je	.L104
	movq	$20, -152(%rbp)
	jmp	.L99
.L104:
	movq	$29, -152(%rbp)
	jmp	.L99
.L88:
	movl	-180(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	testl	%eax, %eax
	jle	.L106
	movq	$22, -152(%rbp)
	jmp	.L99
.L106:
	movq	$3, -152(%rbp)
	jmp	.L99
.L89:
	movq	$23, -152(%rbp)
	jmp	.L99
.L85:
	movl	-196(%rbp), %eax
	cmpl	%eax, -180(%rbp)
	jge	.L109
	movq	$14, -152(%rbp)
	jmp	.L99
.L109:
	movq	$7, -152(%rbp)
	jmp	.L99
.L92:
	movl	-180(%rbp), %eax
	cltq
	movl	-144(%rbp,%rax,4), %eax
	cmpl	%eax, -184(%rbp)
	jl	.L111
	movq	$4, -152(%rbp)
	jmp	.L99
.L111:
	movq	$3, -152(%rbp)
	jmp	.L99
.L82:
	movl	-180(%rbp), %eax
	movl	%eax, -188(%rbp)
	movq	$3, -152(%rbp)
	jmp	.L99
.L96:
	movl	-188(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	leal	-1(%rax), %edx
	movl	-188(%rbp), %eax
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	movq	$8, -152(%rbp)
	jmp	.L99
.L91:
	movl	$999, 352(%rbp)
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -184(%rbp)
	movq	$26, -152(%rbp)
	jmp	.L99
.L77:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -168(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -164(%rbp)
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-176(%rbp), %xmm0
	movl	-196(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -168(%rbp)
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-172(%rbp), %xmm0
	movl	-196(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -164(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-168(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	pxor	%xmm3, %xmm3
	cvtss2sd	-164(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$19, -152(%rbp)
	jmp	.L99
.L83:
	movl	$100, -188(%rbp)
	movl	$0, -180(%rbp)
	movq	$16, -152(%rbp)
	nop
.L99:
	jmp	.L113
.L116:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L115
	call	__stack_chk_fail@PLT
.L115:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	srtf, .-srtf
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
