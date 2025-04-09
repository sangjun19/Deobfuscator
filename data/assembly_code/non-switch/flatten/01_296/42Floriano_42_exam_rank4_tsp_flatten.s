	.file	"42Floriano_42_exam_rank4_tsp_flatten.c"
	.text
	.globl	_TIG_IZ_uS1m_envp
	.bss
	.align 8
	.type	_TIG_IZ_uS1m_envp, @object
	.size	_TIG_IZ_uS1m_envp, 8
_TIG_IZ_uS1m_envp:
	.zero	8
	.globl	_TIG_IZ_uS1m_argc
	.align 4
	.type	_TIG_IZ_uS1m_argc, @object
	.size	_TIG_IZ_uS1m_argc, 4
_TIG_IZ_uS1m_argc:
	.zero	4
	.globl	_TIG_IZ_uS1m_argv
	.align 8
	.type	_TIG_IZ_uS1m_argv, @object
	.size	_TIG_IZ_uS1m_argv, 8
_TIG_IZ_uS1m_argv:
	.zero	8
	.text
	.globl	swap
	.type	swap, @function
swap:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L10
	cmpq	$2, -8(%rbp)
	ja	.L11
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L11
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L5
	movq	$0, -8(%rbp)
	jmp	.L7
.L5:
	movq	$2, -8(%rbp)
	jmp	.L7
.L4:
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	xorl	%eax, %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	xorl	%eax, %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	xorl	%eax, %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	$2, -8(%rbp)
	jmp	.L7
.L11:
	nop
.L7:
	jmp	.L9
.L10:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	swap, .-swap
	.section	.rodata
	.align 8
.LC1:
	.string	"shortest total distance is: %.2f\n"
.LC2:
	.string	"%lf, %lf"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$304, %rsp
	movl	%edi, -276(%rbp)
	movq	%rsi, -288(%rbp)
	movq	%rdx, -296(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_uS1m_envp(%rip)
	nop
.L13:
	movq	$0, _TIG_IZ_uS1m_argv(%rip)
	nop
.L14:
	movl	$0, _TIG_IZ_uS1m_argc(%rip)
	nop
	nop
.L15:
.L16:
#APP
# 90 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-uS1m--0
# 0 "" 2
#NO_APP
	movl	-276(%rbp), %eax
	movl	%eax, _TIG_IZ_uS1m_argc(%rip)
	movq	-288(%rbp), %rax
	movq	%rax, _TIG_IZ_uS1m_argv(%rip)
	movq	-296(%rbp), %rax
	movq	%rax, _TIG_IZ_uS1m_envp(%rip)
	nop
	movq	$2, -248(%rbp)
.L29:
	cmpq	$9, -248(%rbp)
	ja	.L32
	movq	-248(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L32-.L19
	.long	.L32-.L19
	.long	.L24-.L19
	.long	.L23-.L19
	.long	.L32-.L19
	.long	.L22-.L19
	.long	.L32-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L20:
	movsd	.LC0(%rip), %xmm0
	movsd	%xmm0, -256(%rbp)
	leaq	-256(%rbp), %rdi
	leaq	-96(%rbp), %rsi
	leaq	-192(%rbp), %rcx
	movl	-264(%rbp), %edx
	leaq	-240(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	movl	$0, %esi
	movq	%rax, %rdi
	call	find_shortest_path
	movq	-256(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$9, -248(%rbp)
	jmp	.L25
.L23:
	leaq	-96(%rbp), %rdx
	movl	-264(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	(%rdx,%rax), %rcx
	leaq	-192(%rbp), %rdx
	movl	-264(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	stdin(%rip), %rax
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -260(%rbp)
	movq	$7, -248(%rbp)
	jmp	.L25
.L18:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L30
	jmp	.L31
.L22:
	movl	-264(%rbp), %eax
	cltq
	movl	-264(%rbp), %edx
	movl	%edx, -240(%rbp,%rax,4)
	addl	$1, -264(%rbp)
	movq	$3, -248(%rbp)
	jmp	.L25
.L21:
	cmpl	$2, -260(%rbp)
	jne	.L27
	movq	$5, -248(%rbp)
	jmp	.L25
.L27:
	movq	$8, -248(%rbp)
	jmp	.L25
.L24:
	movl	$0, -264(%rbp)
	movq	$3, -248(%rbp)
	jmp	.L25
.L32:
	nop
.L25:
	jmp	.L29
.L31:
	call	__stack_chk_fail@PLT
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	find_shortest_path
	.type	find_shortest_path, @function
find_shortest_path:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movl	%edx, -64(%rbp)
	movq	%rcx, -72(%rbp)
	movq	%r8, -80(%rbp)
	movq	%r9, -88(%rbp)
	movq	$11, -24(%rbp)
.L59:
	cmpq	$18, -24(%rbp)
	ja	.L62
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L47-.L36
	.long	.L46-.L36
	.long	.L45-.L36
	.long	.L62-.L36
	.long	.L62-.L36
	.long	.L62-.L36
	.long	.L44-.L36
	.long	.L63-.L36
	.long	.L42-.L36
	.long	.L41-.L36
	.long	.L40-.L36
	.long	.L39-.L36
	.long	.L62-.L36
	.long	.L38-.L36
	.long	.L62-.L36
	.long	.L37-.L36
	.long	.L62-.L36
	.long	.L62-.L36
	.long	.L35-.L36
	.text
.L35:
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm2
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm1
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm0
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movapd	%xmm2, %xmm3
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	calculate_distance
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movsd	-32(%rbp), %xmm0
	addsd	-16(%rbp), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L48
.L37:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swap
	movl	-60(%rbp), %eax
	leal	1(%rax), %esi
	movq	-88(%rbp), %r8
	movq	-80(%rbp), %rdi
	movq	-72(%rbp), %rcx
	movl	-64(%rbp), %edx
	movq	-56(%rbp), %rax
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rdi
	call	find_shortest_path
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swap
	addl	$1, -36(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L48
.L42:
	movq	-88(%rbp), %rax
	movsd	-32(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movq	$7, -24(%rbp)
	jmp	.L48
.L46:
	movq	-88(%rbp), %rax
	movsd	(%rax), %xmm0
	comisd	-32(%rbp), %xmm0
	jbe	.L61
	movq	$8, -24(%rbp)
	jmp	.L48
.L61:
	movq	$7, -24(%rbp)
	jmp	.L48
.L39:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L48
.L41:
	movl	-36(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm2
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm1
	movl	-36(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm0
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movapd	%xmm2, %xmm3
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	calculate_distance
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movsd	-32(%rbp), %xmm0
	addsd	-8(%rbp), %xmm0
	movsd	%xmm0, -32(%rbp)
	addl	$1, -36(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L48
.L38:
	movl	-64(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -36(%rbp)
	jge	.L52
	movq	$9, -24(%rbp)
	jmp	.L48
.L52:
	movq	$18, -24(%rbp)
	jmp	.L48
.L44:
	movl	$0, -36(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L48
.L40:
	movl	-64(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -60(%rbp)
	jne	.L54
	movq	$6, -24(%rbp)
	jmp	.L48
.L54:
	movq	$0, -24(%rbp)
	jmp	.L48
.L47:
	movl	-60(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L48
.L45:
	movl	-36(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L57
	movq	$15, -24(%rbp)
	jmp	.L48
.L57:
	movq	$7, -24(%rbp)
	jmp	.L48
.L62:
	nop
.L48:
	jmp	.L59
.L63:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	find_shortest_path, .-find_shortest_path
	.globl	calculate_distance
	.type	calculate_distance, @function
calculate_distance:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movsd	%xmm0, -40(%rbp)
	movsd	%xmm1, -48(%rbp)
	movsd	%xmm2, -56(%rbp)
	movsd	%xmm3, -64(%rbp)
	movq	$2, -24(%rbp)
.L70:
	cmpq	$2, -24(%rbp)
	je	.L65
	cmpq	$2, -24(%rbp)
	ja	.L72
	cmpq	$0, -24(%rbp)
	je	.L67
	cmpq	$1, -24(%rbp)
	jne	.L72
	movsd	-48(%rbp), %xmm0
	subsd	-40(%rbp), %xmm0
	movsd	%xmm0, -16(%rbp)
	movsd	-64(%rbp), %xmm0
	subsd	-56(%rbp), %xmm0
	movsd	%xmm0, -8(%rbp)
	movsd	-16(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	%xmm0, %xmm1
	movsd	-8(%rbp), %xmm0
	mulsd	%xmm0, %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L68
.L67:
	movsd	-32(%rbp), %xmm0
	jmp	.L71
.L65:
	movq	$1, -24(%rbp)
	jmp	.L68
.L72:
	nop
.L68:
	jmp	.L70
.L71:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	calculate_distance, .-calculate_distance
	.section	.rodata
	.align 8
.LC0:
	.long	-1
	.long	2146435071
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
